"""G-code backtransform: deformed-space -> original-space.

The forward pipeline is:

    original mesh ──deform_mesh──> deformed mesh ──PrusaSlicer──> G-code(deformed)

The G-code that comes out is in "deformed space" — its horizontal layers
are the curved non-planar layers in the original space. To actually print
on a non-planar 4/5-axis machine we need the G-code coordinates in the
ORIGINAL space, so the print head moves along the curved layer surfaces.

This module does the inverse mapping point-by-point. The deform was:

    new_xy = orig_xy            (XY preserved)
    new_z  = (1 - w) * orig_z   +  w * (depth(orig_xy, orig_z) * dz_per_layer + bed_z)
    w      = clip((orig_z - bed_z) / blend_h, 0, 1)

XY inverts trivially (= deformed XY). Z inverts via 1D root-finding along
the (orig_x = new_x, orig_y = new_y) column of the depth field.

Long G1 moves (>= subdiv_mm in deformed-space length) are subdivided so
straight lines in deformed space follow the curve in original space.
E values (relative, M83) are distributed proportionally; F (feed rate)
is preserved.

Lines that aren't G0/G1 with movement pass through unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import trimesh

from .build_volume import BuildVolume
from .deform import smoothed_depth_field
from .growth import compute_growth
from .mesh_io import load_and_place
from .voxelize import voxelize_solid


_TOKEN_RE = re.compile(r"([A-Z])\s*(-?\d+(?:\.\d+)?)")


class BackTransform:
    """Holds the depth field and parameters needed to invert deform_mesh."""

    def __init__(
        self,
        depth_field: np.ndarray,
        origin: np.ndarray,
        pitch: float,
        dz_per_layer: float,
        bed_z: float = 0.0,
        bed_blend_height: float | None = None,
    ) -> None:
        self.depth_field = depth_field.astype(np.float32)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.pitch = float(pitch)
        self.dz_per_layer = float(dz_per_layer)
        self.bed_z = float(bed_z)
        self.bed_blend_height = float(bed_blend_height) if bed_blend_height is not None else 2.0 * self.pitch

        nx, ny, nz = self.depth_field.shape
        self.shape = (nx, ny, nz)
        # Sampling points (in cell-coordinate Z) for monotone inversion. Each
        # cell center is at (k + 0.5) in cell coords; world z = origin_z + cell * pitch.
        self.k_world_z = self.origin[2] + (np.arange(nz, dtype=np.float32) + 0.5) * self.pitch

    @classmethod
    def from_mesh(
        cls,
        stl_path: str,
        volume_side: float = 250.0,
        pitch: float = 1.0,
        max_tilt_deg: float = 30.0,
        smooth_sigma: float = 2.0,
        depth_method: str = "fmm",
        dz_per_layer: float | None = None,
        bed_blend_height: float | None = None,
    ) -> "BackTransform":
        """Build the same depth field that was used for the forward deform.

        The args must match what was passed to deform_mesh / Viewer when
        the deformed STL was exported, otherwise the inverse won't line up."""
        vol = BuildVolume.cube(volume_side)
        mesh = load_and_place(stl_path, vol)
        vg = voxelize_solid(mesh, pitch=pitch)
        gr = compute_growth(vg, max_tilt_deg=max_tilt_deg)
        # outside_mode='extend' matches deform_mesh's default — that's the
        # field whose forward-applied version was sliced.
        field = smoothed_depth_field(
            gr, sigma=smooth_sigma, method=depth_method, outside_mode="extend"
        )
        if dz_per_layer is None:
            dz_per_layer = pitch
        return cls(
            depth_field=field,
            origin=vg.origin,
            pitch=pitch,
            dz_per_layer=dz_per_layer,
            bed_blend_height=bed_blend_height,
        )

    def _sample_depth_column(self, x: float, y: float) -> np.ndarray:
        """Bilinearly sample the depth field at (x, y) for every k. Returns a
        1D array (nz,) of depth values along the Z column at world (x, y)."""
        nx, ny, _ = self.shape
        u = (x - self.origin[0]) / self.pitch - 0.5
        v = (y - self.origin[1]) / self.pitch - 0.5
        u = float(np.clip(u, 0.0, nx - 1.0001))
        v = float(np.clip(v, 0.0, ny - 1.0001))
        i0 = int(np.floor(u))
        j0 = int(np.floor(v))
        fu = u - i0
        fv = v - j0
        col = (
            self.depth_field[i0,     j0,     :] * (1 - fu) * (1 - fv)
            + self.depth_field[i0 + 1, j0,     :] * fu * (1 - fv)
            + self.depth_field[i0,     j0 + 1, :] * (1 - fu) * fv
            + self.depth_field[i0 + 1, j0 + 1, :] * fu * fv
        )
        return col

    def invert_point(self, x_def: float, y_def: float, z_def: float) -> tuple[float, float, float]:
        """Find (x_orig, y_orig, z_orig) such that the forward deform maps it to
        (x_def, y_def, z_def). XY are identity (deform preserves XY)."""
        # Below the bed: deform is identity (w=0).
        if z_def <= self.bed_z:
            return (x_def, y_def, z_def)

        # World->cell column at (x_def, y_def). The forward map at fixed (x, y) is
        # f(z) = (1 - w(z)) * z + w(z) * (depth_at_z * dz_per_layer + bed_z).
        # f is monotone in z for our use cases (curved layers are not folded back).
        col = self._sample_depth_column(x_def, y_def)
        k_world_z = self.k_world_z  # original z at each cell center

        # Forward at each cell-center z.
        blend = self.bed_blend_height
        bed = self.bed_z
        if blend > 0:
            w = np.clip((k_world_z - bed) / blend, 0.0, 1.0)
        else:
            w = np.ones_like(k_world_z)
        z_target = col * self.dz_per_layer + bed
        f = (1.0 - w) * k_world_z + w * z_target  # deformed Z for each cell-center original Z

        # Binary search for z_def in (sorted-ish) f. f should be monotone-increasing.
        if z_def <= f[0]:
            return (x_def, y_def, float(k_world_z[0]))
        if z_def >= f[-1]:
            return (x_def, y_def, float(k_world_z[-1]))

        # Find first index where f[k] >= z_def.
        k = int(np.searchsorted(f, z_def))
        # Linear interp between (f[k-1], k_world_z[k-1]) and (f[k], k_world_z[k]).
        f0, f1 = float(f[k - 1]), float(f[k])
        z0, z1 = float(k_world_z[k - 1]), float(k_world_z[k])
        if f1 == f0:
            z_orig = 0.5 * (z0 + z1)
        else:
            t = (z_def - f0) / (f1 - f0)
            z_orig = z0 + t * (z1 - z0)
        return (x_def, y_def, float(z_orig))


def _parse_xyzef(rest: str) -> dict[str, float]:
    """Parse the param tokens after a G0/G1 keyword into a dict of {letter: value}."""
    out: dict[str, float] = {}
    for letter, val in _TOKEN_RE.findall(rest):
        if letter in "XYZEF":
            out[letter] = float(val)
    return out


def _format_xyz(x: float, y: float, z: float) -> str:
    """G-code-friendly XYZ formatting (3 decimal places)."""
    return f"X{x:.3f} Y{y:.3f} Z{z:.3f}"


def backtransform_gcode_file(
    input_path: str | Path,
    output_path: str | Path,
    bt: BackTransform,
    subdiv_mm: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Process a G-code file end-to-end. Returns stats dict.

    Move handling:
      - G0 / G1 with XYZ → endpoint inverted; if the deformed-space move is
        longer than `subdiv_mm`, the segment is split into pieces of about
        that length so the curved original-space path is followed.
      - E (assumed relative, M83) is distributed across subdivided pieces
        proportionally by deformed-space arc length.
      - F (feed) is kept on the first emitted line of a multi-piece move.
      - Other commands (M, G92, comments, etc.) pass through unchanged.

    State carried across lines: current absolute X, Y, Z; current feed; the
    'extrusion relative' flag is parsed (M82/M83) and logged but the writer
    assumes relative (M83) — PrusaSlicer's default mode."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    stats = {
        "lines_in": 0,
        "lines_out": 0,
        "moves_in": 0,
        "moves_out": 0,
        "z_min": float("inf"),
        "z_max": float("-inf"),
        "z_orig_min": float("inf"),
        "z_orig_max": float("-inf"),
    }

    cur_x = cur_y = cur_z = 0.0
    cur_f: float | None = None
    e_relative = True
    seen_first_move = False

    with in_path.open("r", encoding="utf-8", errors="replace") as fi, \
         out_path.open("w", encoding="utf-8", newline="\n") as fo:

        fo.write(
            "; backtransformed by vff/backtransform.py — "
            "deformed-space XYZ inverted to original (non-planar) space\n"
        )

        for raw in fi:
            stats["lines_in"] += 1
            line = raw.rstrip("\r\n")
            stripped = line.lstrip()
            if not stripped or stripped.startswith(";"):
                fo.write(line + "\n"); stats["lines_out"] += 1
                continue

            # Cheap leading-token grab (G1, M104, etc.). Comments after the
            # command are preserved by appending tail.
            head, _, tail = line.partition(";")
            head_tokens = head.split()
            if not head_tokens:
                fo.write(line + "\n"); stats["lines_out"] += 1
                continue
            cmd = head_tokens[0].upper()

            if cmd in ("M82",):
                e_relative = False
                fo.write(line + "\n"); stats["lines_out"] += 1
                continue
            if cmd in ("M83",):
                e_relative = True
                fo.write(line + "\n"); stats["lines_out"] += 1
                continue

            if cmd in ("G0", "G1", "G00", "G01"):
                params = _parse_xyzef(" ".join(head_tokens[1:]))
                new_x = params.get("X", cur_x)
                new_y = params.get("Y", cur_y)
                new_z = params.get("Z", cur_z)
                e_val = params.get("E", None)
                f_val = params.get("F", None)
                stats["moves_in"] += 1

                # If only F changed and no XYZ/E move, just echo.
                if (
                    new_x == cur_x and new_y == cur_y and new_z == cur_z
                    and e_val is None
                ):
                    fo.write(line + "\n"); stats["lines_out"] += 1
                    if f_val is not None:
                        cur_f = f_val
                    continue

                # Track deformed-space Z range we saw (for the stats line).
                if seen_first_move:
                    stats["z_min"] = min(stats["z_min"], new_z)
                    stats["z_max"] = max(stats["z_max"], new_z)
                else:
                    stats["z_min"] = new_z
                    stats["z_max"] = new_z
                    seen_first_move = True

                # Decide subdivision count.
                seg_len = float(np.hypot(np.hypot(new_x - cur_x, new_y - cur_y), new_z - cur_z))
                if subdiv_mm > 0 and seg_len > subdiv_mm:
                    n_pieces = max(1, int(np.ceil(seg_len / subdiv_mm)))
                else:
                    n_pieces = 1

                # Walk along the segment in deformed space, invert each
                # piece's endpoint, distribute E (assumed relative).
                if cmd in ("G0", "G00"):
                    out_cmd = "G0"
                else:
                    out_cmd = "G1"

                first_piece = True
                for piece in range(1, n_pieces + 1):
                    t = piece / n_pieces
                    px = cur_x + (new_x - cur_x) * t
                    py = cur_y + (new_y - cur_y) * t
                    pz = cur_z + (new_z - cur_z) * t
                    ox, oy, oz = bt.invert_point(px, py, pz)

                    parts = [out_cmd, _format_xyz(ox, oy, oz)]
                    if e_val is not None:
                        # Relative E: split proportionally.
                        if e_relative:
                            de = e_val / n_pieces
                            parts.append(f"E{de:.5f}")
                        else:
                            # Absolute E: linear interp from previous cumulative
                            # E... we don't track previous absolute E here, so
                            # emit the final value on the LAST piece only and
                            # skip E on intermediates. Not common with PrusaSlicer.
                            if piece == n_pieces:
                                parts.append(f"E{e_val:.5f}")
                    if f_val is not None and first_piece:
                        parts.append(f"F{f_val:g}")
                        cur_f = f_val

                    fo.write(" ".join(parts))
                    if tail:
                        fo.write(" ;" + tail)
                    fo.write("\n")
                    stats["lines_out"] += 1
                    stats["moves_out"] += 1
                    first_piece = False

                    stats["z_orig_min"] = min(stats["z_orig_min"], oz)
                    stats["z_orig_max"] = max(stats["z_orig_max"], oz)

                cur_x = new_x
                cur_y = new_y
                cur_z = new_z
                continue

            if cmd == "G92":
                # Set position. Parse X/Y/Z/E and update our tracked state so
                # subsequent moves are relative to the right origin.
                params = _parse_xyzef(" ".join(head_tokens[1:]))
                if "X" in params: cur_x = params["X"]
                if "Y" in params: cur_y = params["Y"]
                if "Z" in params: cur_z = params["Z"]
                fo.write(line + "\n"); stats["lines_out"] += 1
                continue

            # Everything else: pass through.
            fo.write(line + "\n"); stats["lines_out"] += 1

    if verbose:
        print(
            f"[backtransform] {in_path.name} -> {out_path.name}: "
            f"{stats['lines_in']:,} -> {stats['lines_out']:,} lines  "
            f"({stats['moves_in']:,} moves -> {stats['moves_out']:,}; "
            f"subdiv≈{stats['moves_out']/max(1,stats['moves_in']):.1f}x)"
        )
        print(
            f"[backtransform] deformed Z: [{stats['z_min']:.3f}, {stats['z_max']:.3f}] mm  "
            f"→ original Z: [{stats['z_orig_min']:.3f}, {stats['z_orig_max']:.3f}] mm"
        )
    return stats
