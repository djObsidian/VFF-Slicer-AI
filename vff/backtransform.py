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


_TOKEN_RE = re.compile(r"([A-Z])\s*(-?(?:\d+\.\d*|\.\d+|\d+))")


def quick_gcode_xy_bounds(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Fast scan for the XY range of *extrusion* moves only. Returns
    (xy_min[2], xy_max[2]).

    Used to align the depth field with the slicer's actual model placement.
    We deliberately ignore G0 travels and G1 lines without a positive E
    delta — those go off to corners (X=0 home, Y=0 prime, purge tower at
    the back, etc.) and would inflate the bbox so its centre no longer
    matches where the part actually prints. Tracks G92 resets and the
    M82/M83 (absolute/relative E) state to compute the delta correctly.
    """
    xy_min = np.array([np.inf, np.inf], dtype=np.float64)
    xy_max = np.array([-np.inf, -np.inf], dtype=np.float64)
    in_path = Path(path)
    cur_x = cur_y = 0.0
    cur_e = 0.0
    e_relative = True
    with in_path.open("r", encoding="utf-8", errors="replace") as fi:
        for raw in fi:
            line = raw.lstrip()
            if not line:
                continue
            head, _semi, _tail = line.partition(";")
            tokens = head.split()
            if not tokens:
                continue
            cmd = tokens[0].upper()
            if cmd == "M82":
                e_relative = False
                continue
            if cmd == "M83":
                e_relative = True
                continue
            if cmd == "G92":
                for letter, val in _TOKEN_RE.findall(head):
                    if letter == "X": cur_x = float(val)
                    elif letter == "Y": cur_y = float(val)
                    elif letter == "E": cur_e = float(val)
                continue
            if cmd not in ("G1", "G01"):
                # G0/travel ignored — only extrusion-bearing G1 contributes to bbox.
                # Still need to track XY if it's a G0 with movement, so subsequent
                # G1 has correct start XY; but for bbox purposes we exclude G0.
                if cmd in ("G0", "G00"):
                    for letter, val in _TOKEN_RE.findall(head):
                        if letter == "X": cur_x = float(val)
                        elif letter == "Y": cur_y = float(val)
                continue

            new_x, new_y = cur_x, cur_y
            e_val = None
            for letter, val in _TOKEN_RE.findall(head):
                if letter == "X": new_x = float(val)
                elif letter == "Y": new_y = float(val)
                elif letter == "E": e_val = float(val)
            # Compute extrusion delta.
            de = 0.0
            if e_val is not None:
                if e_relative:
                    de = e_val
                else:
                    de = e_val - cur_e
                    cur_e = e_val
            # Only contribute to bbox if extrusion actually happened.
            if de > 1e-9:
                if new_x < xy_min[0]: xy_min[0] = new_x
                if new_x > xy_max[0]: xy_max[0] = new_x
                if new_y < xy_min[1]: xy_min[1] = new_y
                if new_y > xy_max[1]: xy_max[1] = new_y
            cur_x, cur_y = new_x, new_y
    return xy_min, xy_max


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
        xy_center: tuple[float, float] | None = None,
    ) -> "BackTransform":
        """Build the same depth field that was used for the forward deform.

        If `xy_center` is given, the mesh is translated so its XY bbox is
        centred on that point (instead of the build-volume centre that
        load_and_place would pick). This is essential for the G-code
        transform path: the slicer (PrusaSlicer, Cura, etc.) places the
        model at its own bed centre, which won't match our build_volume
        centre unless they happen to be the same. Pass the gcode-derived
        XY centre here so the depth field lines up with gcode coords.
        Z is always normalised so the mesh's Z_min sits at 0 (bed)."""
        import trimesh
        mesh = trimesh.load(stl_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Not a single mesh: {type(mesh).__name__}")

        if xy_center is not None:
            mesh_xy_c = 0.5 * (mesh.bounds[0, :2] + mesh.bounds[1, :2])
            dx = float(xy_center[0]) - float(mesh_xy_c[0])
            dy = float(xy_center[1]) - float(mesh_xy_c[1])
            dz = -float(mesh.bounds[0, 2])  # bed-align
            mesh.apply_translation([dx, dy, dz])
        else:
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
        """Single-point convenience wrapper around invert_points_batch."""
        out = self.invert_points_batch(np.array([[x_def, y_def, z_def]], dtype=np.float64))
        return (float(out[0, 0]), float(out[0, 1]), float(out[0, 2]))

    def forward_points_batch(self, xyz_orig: np.ndarray) -> np.ndarray:
        """Vectorised forward deform: (N, 3) ORIGINAL-space → (N, 3) DEFORMED.

        This is the actual deform_mesh map (extend mode on the depth field):

            new_xy = orig_xy
            w      = clip((z - bed_z) / blend, 0, 1)
            new_z  = (1 - w) * z + w * (depth(x, y, z) * dz_per_layer + bed_z)

        Used to convert a planar slicer's G-code (flat layers in the
        deformed-space mesh that was sliced) into non-planar G-code that
        follows the depth field's curved layers in the original mesh."""
        from .deform import _sample_trilinear
        xyz = np.asarray(xyz_orig, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz_orig must be (N, 3)")
        if xyz.shape[0] == 0:
            return xyz.copy()
        depth = _sample_trilinear(self.depth_field, self.origin, self.pitch, xyz).astype(np.float64)
        z = xyz[:, 2]
        blend = self.bed_blend_height
        bed = self.bed_z
        if blend > 0:
            w = np.clip((z - bed) / blend, 0.0, 1.0)
        else:
            w = np.ones_like(z)
        z_target = depth * self.dz_per_layer + bed
        z_new = (1.0 - w) * z + w * z_target
        out = xyz.copy()
        out[:, 2] = z_new
        return out

    def invert_points_batch(self, xyz_def: np.ndarray) -> np.ndarray:
        """Vectorised inverse: takes (N, 3) deformed-space points, returns
        (N, 3) original-space points. ~100× faster than calling invert_point
        in a Python loop on big G-code files."""
        xyz_def = np.asarray(xyz_def, dtype=np.float64)
        if xyz_def.ndim != 2 or xyz_def.shape[1] != 3:
            raise ValueError("xyz_def must be (N, 3)")
        N = xyz_def.shape[0]
        if N == 0:
            return xyz_def.copy()
        nx, ny, nz = self.shape

        x = xyz_def[:, 0]
        y = xyz_def[:, 1]
        z_def = xyz_def[:, 2]

        # Bilinear sample column at (x, y).
        u = (x - self.origin[0]) / self.pitch - 0.5
        v = (y - self.origin[1]) / self.pitch - 0.5
        u = np.clip(u, 0.0, nx - 1.0001)
        v = np.clip(v, 0.0, ny - 1.0001)
        i0 = np.floor(u).astype(np.int64)
        j0 = np.floor(v).astype(np.int64)
        fu = (u - i0).astype(np.float32)
        fv = (v - j0).astype(np.float32)

        # Column samples at each of the 4 surrounding (i, j) cells, full nz.
        c00 = self.depth_field[i0,     j0,     :]
        c10 = self.depth_field[i0 + 1, j0,     :]
        c01 = self.depth_field[i0,     j0 + 1, :]
        c11 = self.depth_field[i0 + 1, j0 + 1, :]
        col = (
            c00 * ((1 - fu) * (1 - fv))[:, None]
            + c10 * (fu       * (1 - fv))[:, None]
            + c01 * ((1 - fu) * fv      )[:, None]
            + c11 * (fu       * fv      )[:, None]
        )  # (N, nz)

        # Forward f(z) at each cell-center z, per row.
        k_world_z = self.k_world_z.astype(np.float32)  # (nz,)
        blend = self.bed_blend_height
        bed = self.bed_z
        if blend > 0:
            w = np.clip((k_world_z - bed) / blend, 0.0, 1.0)
        else:
            w = np.ones_like(k_world_z)
        z_target = col * self.dz_per_layer + bed
        f = (1.0 - w)[None, :] * k_world_z[None, :] + w[None, :] * z_target  # (N, nz)

        z_def_f = z_def.astype(np.float32)
        above = f > z_def_f[:, None]
        any_above = above.any(axis=1)
        # k = first index where f exceeds z_def (per row). 0 means below grid.
        k_first = np.argmax(above, axis=1)  # (N,)

        # Defaults — for points above the grid take the top.
        z_orig = np.full(N, k_world_z[-1], dtype=np.float64)

        # Below f[0]: identity (bed-blend region, w≈0).
        below = z_def_f <= f[:, 0]
        z_orig[below] = z_def[below]

        # Linear interp between f[k-1] and f[k] for the rest.
        interp_mask = any_above & ~below & (k_first > 0)
        if interp_mask.any():
            k_use = k_first[interp_mask]
            f0 = np.take_along_axis(f[interp_mask], (k_use - 1)[:, None], axis=1)[:, 0]
            f1 = np.take_along_axis(f[interp_mask],  k_use[:, None],      axis=1)[:, 0]
            z0 = k_world_z[k_use - 1]
            z1 = k_world_z[k_use]
            df = f1 - f0
            df_safe = np.where(df > 1e-9, df, 1.0)
            t = (z_def_f[interp_mask] - f0) / df_safe
            z_orig[interp_mask] = z0 + t * (z1 - z0)

        return np.column_stack([x, y, z_orig.astype(np.float64)])


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


def _collect_gcode_units(in_path: Path, subdiv_mm: float) -> tuple[list, np.ndarray, dict]:
    """Pass 1: parse G-code, return (units, xyz_def, stats).

    units: list of either
      ('raw', "<text without trailing newline>")
      ('move', cmd, e_val_or_None, f_val_or_None, tail_or_None, n_pieces,
       start_idx_in_xyz_def, e_relative)
    xyz_def: (M, 3) float64 array of subdivided move endpoints (input-space).
    """
    units: list = []
    xyz_def_chunks: list[list[tuple[float, float, float]]] = []
    chunk_buf: list[tuple[float, float, float]] = []

    def _flush_chunk():
        if chunk_buf:
            xyz_def_chunks.append(chunk_buf.copy())
            chunk_buf.clear()

    cur_x = cur_y = cur_z = 0.0
    e_relative = True
    stats = {
        "lines_in": 0,
        "moves_in": 0,
        "moves_out": 0,
        "z_min": float("inf"),
        "z_max": float("-inf"),
    }

    with in_path.open("r", encoding="utf-8", errors="replace") as fi:
        for raw in fi:
            stats["lines_in"] += 1
            line = raw.rstrip("\r\n")
            stripped = line.lstrip()
            if not stripped or stripped.startswith(";"):
                units.append(("raw", line))
                continue

            head, _semi, tail = line.partition(";")
            head_tokens = head.split()
            if not head_tokens:
                units.append(("raw", line))
                continue
            cmd = head_tokens[0].upper()

            if cmd == "M82":
                e_relative = False; units.append(("raw", line)); continue
            if cmd == "M83":
                e_relative = True;  units.append(("raw", line)); continue

            if cmd in ("G0", "G1", "G00", "G01"):
                params = _parse_xyzef(" ".join(head_tokens[1:]))
                new_x = params.get("X", cur_x)
                new_y = params.get("Y", cur_y)
                new_z = params.get("Z", cur_z)
                e_val = params.get("E", None)
                f_val = params.get("F", None)
                stats["moves_in"] += 1

                if new_x == cur_x and new_y == cur_y and new_z == cur_z and e_val is None:
                    units.append(("raw", line))
                    continue

                if stats["z_min"] == float("inf"):
                    stats["z_min"] = stats["z_max"] = new_z
                else:
                    if new_z < stats["z_min"]: stats["z_min"] = new_z
                    if new_z > stats["z_max"]: stats["z_max"] = new_z

                seg_len = float(np.hypot(np.hypot(new_x - cur_x, new_y - cur_y), new_z - cur_z))
                n_pieces = max(1, int(np.ceil(seg_len / subdiv_mm))) if (subdiv_mm > 0 and seg_len > subdiv_mm) else 1

                out_cmd = "G0" if cmd in ("G0", "G00") else "G1"
                start_idx = sum(len(c) for c in xyz_def_chunks) + len(chunk_buf)
                for piece in range(1, n_pieces + 1):
                    t = piece / n_pieces
                    chunk_buf.append((
                        cur_x + (new_x - cur_x) * t,
                        cur_y + (new_y - cur_y) * t,
                        cur_z + (new_z - cur_z) * t,
                    ))
                    if len(chunk_buf) >= 65536:
                        _flush_chunk()

                units.append(("move", out_cmd, e_val, f_val, tail if _semi else None,
                              n_pieces, start_idx, e_relative))
                stats["moves_out"] += n_pieces
                cur_x, cur_y, cur_z = new_x, new_y, new_z
                continue

            if cmd == "G92":
                params = _parse_xyzef(" ".join(head_tokens[1:]))
                if "X" in params: cur_x = params["X"]
                if "Y" in params: cur_y = params["Y"]
                if "Z" in params: cur_z = params["Z"]
                units.append(("raw", line))
                continue

            units.append(("raw", line))

    _flush_chunk()

    if not xyz_def_chunks:
        xyz_def = np.zeros((0, 3), dtype=np.float64)
    else:
        xyz_def = np.concatenate([np.asarray(c, dtype=np.float64) for c in xyz_def_chunks], axis=0)

    return units, xyz_def, stats


def _write_clipped_gcode(
    out_path: Path,
    units: list,
    xyz_out: np.ndarray,
    inside_mask: np.ndarray,
    header: str,
) -> tuple[int, int, int, float, float]:
    """Pass 3 (clip variant): per-piece, emit G1+E if `inside_mask[piece]` is
    true, else G0 with no E (path kept, extrusion dropped).

    For absolute-E (M82) moves: PrusaSlicer's gcode is usually relative (M83),
    but if we encounter absolute, we still emit the final absolute E only when
    the LAST piece is inside — otherwise the printer's E position would
    desync. We compensate by emitting the last-inside-piece's E instead.

    Returns (lines_written, n_kept, n_dropped, e_kept_mm, e_dropped_mm).
    """
    lines_out = 0
    n_kept = 0
    n_dropped = 0
    e_kept = 0.0
    e_dropped = 0.0
    with out_path.open("w", encoding="utf-8", newline="\n") as fo:
        fo.write(header)
        lines_out += 1
        for u in units:
            if u[0] == "raw":
                fo.write(u[1] + "\n")
                lines_out += 1
                continue
            _, out_cmd, e_val, f_val, tail, n_pieces, start_idx, e_rel = u
            # For absolute E: find the last inside-piece so we can emit the
            # cumulative E there. If no piece is inside, the move skips E.
            last_inside_piece = -1
            if out_cmd == "G1" and e_val is not None and not e_rel:
                for piece in range(n_pieces - 1, -1, -1):
                    if inside_mask[start_idx + piece]:
                        last_inside_piece = piece
                        break
            for piece in range(n_pieces):
                pi = start_idx + piece
                ox, oy, oz = xyz_out[pi]
                piece_inside = bool(inside_mask[pi])
                e_str = None
                if out_cmd == "G0":
                    cmd = "G0"
                elif piece_inside:
                    cmd = "G1"
                    if e_val is not None:
                        if e_rel:
                            piece_e = e_val / n_pieces
                            e_str = f"E{piece_e:.5f}"
                            e_kept += piece_e
                        elif piece == last_inside_piece:
                            e_str = f"E{e_val:.5f}"
                    n_kept += 1
                else:
                    cmd = "G0"
                    n_dropped += 1
                    if e_val is not None and e_rel:
                        e_dropped += e_val / n_pieces
                parts = [cmd, f"X{ox:.3f} Y{oy:.3f} Z{oz:.3f}"]
                if e_str:
                    parts.append(e_str)
                if f_val is not None and piece == 0:
                    parts.append(f"F{f_val:g}")
                fo.write(" ".join(parts))
                if tail is not None:
                    fo.write(" ;" + tail)
                fo.write("\n")
                lines_out += 1
    return lines_out, n_kept, n_dropped, e_kept, e_dropped


def clip_gcode_file(
    input_path: str | Path,
    output_path: str | Path,
    clip_stl_path: str | Path,
    *,
    xy_center: tuple[float, float] | None = None,
    subdiv_mm: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Clip G-code extrusion to a target STL's interior.

    For each subdivided gcode piece, test whether the piece's endpoint lies
    inside `clip_stl_path` (Z-normalised to Z_min=0, XY-translated so its XY
    bbox centre matches `xy_center`). Pieces inside → emit as G1 with their
    share of E. Pieces outside → emit as G0 (no extrusion). Travel-only moves
    (G0) pass through.

    Use case: 'gcode was sliced for a flat-bottom variant (target + filled
    base/slab); I want only the extrusions that fall inside the real
    curved-bottom target to make it to the printer.' Material that PrusaSlicer
    deposited into the filled-base region is dropped. The printer's head path
    is preserved (still moves through the dropped region as travel), so XYZ
    sequencing is unchanged — only E is removed for outside pieces.

    Caveats:
      - Mesh must be closed/manifold for `.contains()` to be reliable.
      - For real printing this can leave material unsupported (e.g. blade
        tips at high Z without lower layers under them). This mode is
        primarily for VISUAL VERIFICATION that the deformation pipeline
        produces the right shape — for actual prints you still need
        supports/transitions.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    import time as _time
    t0 = _time.perf_counter()

    target = _build_target_mesh(clip_stl_path, xy_center)
    tgt_bounds = target.bounds.tolist()

    units, xyz_def, stats = _collect_gcode_units(in_path, subdiv_mm)
    t1 = _time.perf_counter()

    if xyz_def.shape[0] == 0:
        inside = np.zeros(0, dtype=bool)
    else:
        inside = np.asarray(target.contains(xyz_def), dtype=bool)
    t2 = _time.perf_counter()

    header = (
        "; clip-to-mesh by vff/backtransform.py — "
        f"extrusion dropped where outside {Path(clip_stl_path).name}\n"
    )
    lines_out, n_kept, n_dropped, e_kept, e_dropped = _write_clipped_gcode(
        out_path, units, xyz_def, inside, header,
    )
    t3 = _time.perf_counter()

    stats["lines_out"] = lines_out
    stats["n_pts"] = int(xyz_def.shape[0])
    stats["n_pieces_kept"] = n_kept
    stats["n_pieces_dropped"] = n_dropped
    stats["e_kept_mm"] = e_kept
    stats["e_dropped_mm"] = e_dropped
    stats["target_bounds"] = tgt_bounds
    stats["t_parse_s"] = t1 - t0
    stats["t_contains_s"] = t2 - t1
    stats["t_write_s"] = t3 - t2

    if verbose:
        print(
            f"[clip] {in_path.name} -> {out_path.name}: "
            f"{stats['lines_in']:,} -> {stats['lines_out']:,} lines "
            f"({stats['moves_in']:,} moves -> {stats['moves_out']:,} pieces; "
            f"subdiv≈{stats['moves_out']/max(1,stats['moves_in']):.1f}x)"
        )
        print(
            f"[clip] target bounds: X[{tgt_bounds[0][0]:.1f},{tgt_bounds[1][0]:.1f}] "
            f"Y[{tgt_bounds[0][1]:.1f},{tgt_bounds[1][1]:.1f}] "
            f"Z[{tgt_bounds[0][2]:.2f},{tgt_bounds[1][2]:.2f}] mm"
        )
        n_total = max(1, n_kept + n_dropped)
        print(
            f"[clip] pieces: kept {n_kept:,} ({100*n_kept/n_total:.1f}%) / "
            f"dropped {n_dropped:,} ({100*n_dropped/n_total:.1f}%); "
            f"extrusion kept {e_kept:.0f} mm / dropped {e_dropped:.0f} mm"
        )
        print(
            f"[clip] timing: parse {stats['t_parse_s']:.2f}s "
            f"contains {stats['t_contains_s']:.2f}s ({stats['n_pts']:,} pts) "
            f"write {stats['t_write_s']:.2f}s"
        )
    return stats


def _write_transformed_gcode(out_path: Path, units: list, xyz_out: np.ndarray, header: str) -> int:
    """Pass 3: stream-write the unit list, reading move endpoints from xyz_out."""
    lines_out = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as fo:
        fo.write(header)
        lines_out += 1
        for u in units:
            if u[0] == "raw":
                fo.write(u[1] + "\n")
                lines_out += 1
                continue
            _, out_cmd, e_val, f_val, tail, n_pieces, start_idx, e_rel = u
            for piece in range(n_pieces):
                ox, oy, oz = xyz_out[start_idx + piece]
                parts = [out_cmd, f"X{ox:.3f} Y{oy:.3f} Z{oz:.3f}"]
                if e_val is not None:
                    if e_rel:
                        parts.append(f"E{(e_val / n_pieces):.5f}")
                    elif piece == n_pieces - 1:
                        parts.append(f"E{e_val:.5f}")
                if f_val is not None and piece == 0:
                    parts.append(f"F{f_val:g}")
                fo.write(" ".join(parts))
                if tail is not None:
                    fo.write(" ;" + tail)
                fo.write("\n")
                lines_out += 1
    return lines_out


def backtransform_gcode_file(
    input_path: str | Path,
    output_path: str | Path,
    bt: BackTransform,
    subdiv_mm: float = 0.5,
    n_jobs: int = -1,
    verbose: bool = True,
    direction: str = "forward",
) -> dict:
    """Three-pass batched G-code transform using a BackTransform's depth field.

    direction:
      - "forward"  (default): apply deform_mesh's map to each G-code point.
        Use case: planar-slicer output for a flat-bottom (or otherwise
        slicer-friendly) mesh → non-planar G-code that follows the depth
        field's curved layers when printed.
      - "inverse" : apply deform_mesh's inverse map. Use case: planar
        slicer output for a *deformed* mesh (one produced by deform_mesh) →
        G-code that prints those flat slicer layers as curved layers in
        the ORIGINAL (un-deformed) mesh's coordinate system.

    `n_jobs`: -1 (default) uses all CPU cores via multiprocessing.Pool.
              0 / 1 runs in-process (avoids fork overhead on small files).
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    import time as _time
    t0 = _time.perf_counter()

    units, xyz_def, stats = _collect_gcode_units(in_path, subdiv_mm)

    t1 = _time.perf_counter()

    n_pts = xyz_def.shape[0]
    if direction not in ("forward", "inverse"):
        raise ValueError(f"direction must be 'forward' or 'inverse', got {direction!r}")
    if n_pts == 0:
        xyz_orig = xyz_def.copy()
    else:
        # Multiprocessing only pays off above a few million points — on Windows
        # the spawn-context startup (re-import scipy / vff / pickle the depth
        # field for every worker) costs several seconds, which a vectorised
        # NumPy single-pass beats below ~5 M pts. Override with --jobs N to
        # force the parallel path anyway.
        _MP_AUTO_THRESHOLD = 5_000_000
        if n_jobs in (0, 1) or (n_jobs == -1 and n_pts < _MP_AUTO_THRESHOLD):
            if direction == "forward":
                xyz_orig = bt.forward_points_batch(xyz_def)
            else:
                xyz_orig = bt.invert_points_batch(xyz_def)
        else:
            import multiprocessing as _mp
            try:
                n_workers = _mp.cpu_count() if n_jobs == -1 else int(n_jobs)
            except NotImplementedError:
                n_workers = 4
            n_workers = max(1, min(n_workers, 64, max(1, n_pts // 4000)))
            chunk_size = (n_pts + n_workers - 1) // n_workers
            args = [
                (direction, xyz_def[i:i + chunk_size], bt.depth_field, bt.origin, bt.pitch,
                 bt.dz_per_layer, bt.bed_z, bt.bed_blend_height)
                for i in range(0, n_pts, chunk_size)
            ]
            with _mp.get_context("spawn").Pool(n_workers) as pool:
                results = pool.starmap(_transform_chunk_worker, args)
            xyz_orig = np.concatenate(results, axis=0)

    t2 = _time.perf_counter()

    z_orig_min = float("inf")
    z_orig_max = float("-inf")
    if n_pts:
        z_orig_min = float(xyz_orig[:, 2].min())
        z_orig_max = float(xyz_orig[:, 2].max())

    header = (
        "; backtransformed by vff/backtransform.py — "
        "deformed-space XYZ inverted to original (non-planar) space\n"
    )
    stats["lines_out"] = _write_transformed_gcode(out_path, units, xyz_orig, header)

    t3 = _time.perf_counter()

    stats["z_orig_min"] = z_orig_min
    stats["z_orig_max"] = z_orig_max
    stats["t_parse_s"] = t1 - t0
    stats["t_invert_s"] = t2 - t1
    stats["t_write_s"] = t3 - t2
    stats["n_invert_pts"] = int(n_pts)

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
        print(
            f"[backtransform] timing: parse {stats['t_parse_s']:.2f}s  "
            f"invert {stats['t_invert_s']:.2f}s ({stats['n_invert_pts']:,} pts)  "
            f"write {stats['t_write_s']:.2f}s"
        )
    return stats


def _build_target_mesh(
    target_stl_path: str | Path,
    xy_center: tuple[float, float] | None = None,
) -> trimesh.Trimesh:
    """Load target STL, Z-normalise so Z_min = 0, optionally XY-translate so
    its XY bbox centre matches xy_center."""
    target = trimesh.load(str(target_stl_path), force="mesh")
    if not isinstance(target, trimesh.Trimesh):
        raise ValueError(f"Not a single mesh: {type(target).__name__}")
    target.apply_translation([0.0, 0.0, -float(target.bounds[0, 2])])
    if xy_center is not None:
        mxy = 0.5 * (target.bounds[0, :2] + target.bounds[1, :2])
        target.apply_translation([
            float(xy_center[0]) - float(mxy[0]),
            float(xy_center[1]) - float(mxy[1]),
            0.0,
        ])
    return target


def _surface_heights_bottom(target: trimesh.Trimesh, xy_points: np.ndarray) -> np.ndarray:
    """Ray-cast from below into `target` at each (X, Y); return per-point Z of
    the lowest surface hit (the target's bottom face at that column).
    Returns 0 for (X, Y) outside the target's XY footprint (ray misses)."""
    xy_points = np.asarray(xy_points, dtype=np.float64)
    N = xy_points.shape[0]
    if N == 0:
        return np.zeros(0, dtype=np.float64)
    z_start = float(target.bounds[0, 2]) - 1.0
    origins = np.column_stack([
        xy_points[:, 0], xy_points[:, 1], np.full(N, z_start, dtype=np.float64),
    ])
    directions = np.tile([0.0, 0.0, 1.0], (N, 1))
    locations, ray_indices, _ = target.ray.intersects_location(
        origins, directions, multiple_hits=False,
    )
    H = np.zeros(N, dtype=np.float64)
    if len(ray_indices) > 0:
        H[np.asarray(ray_indices, dtype=np.int64)] = locations[:, 2]
    return H


def surface_offset_gcode_file(
    input_path: str | Path,
    output_path: str | Path,
    target_stl_path: str | Path,
    *,
    xy_center: tuple[float, float] | None = None,
    subdiv_mm: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Conform G-code Z to a target STL's bottom surface.

    For each gcode point (X, Y, Z_planar), find H(X, Y) — the Z of
    `target_stl_path`'s lowest surface at that XY column (ray-cast from
    below) — and output (X, Y, Z_planar + H). The target STL is
    Z-normalised so its Z_min = 0 (bed), and XY-translated so its XY-bbox
    centre matches `xy_center` (typically the gcode's extrusion-XY centre).

    Use case: 'print target_stl shape using gcode that was sliced from a
    flat-bottomed variant (target + flat slab base)'. Unlike the
    depth-field forward map (which compresses Z by depth/depth_max),
    this preserves slab thickness, blade thickness and hub height
    everywhere — every column is rigidly Z-shifted by H(X, Y), nothing
    is squashed.

    For XY points that fall outside the target's footprint, H = 0
    (point passes through unchanged)."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    import time as _time
    t0 = _time.perf_counter()

    target = _build_target_mesh(target_stl_path, xy_center)
    tgt_bounds = target.bounds.tolist()

    units, xyz_def, stats = _collect_gcode_units(in_path, subdiv_mm)
    t1 = _time.perf_counter()

    if xyz_def.shape[0] == 0:
        xyz_out = xyz_def.copy()
        H = np.zeros(0, dtype=np.float64)
    else:
        H = _surface_heights_bottom(target, xyz_def[:, :2])
        xyz_out = xyz_def.copy()
        xyz_out[:, 2] = xyz_def[:, 2] + H
    t2 = _time.perf_counter()

    header = (
        "; conform-to-surface by vff/backtransform.py — "
        f"Z lifted by bottom surface of {Path(target_stl_path).name}\n"
    )
    stats["lines_out"] = _write_transformed_gcode(out_path, units, xyz_out, header)
    t3 = _time.perf_counter()

    z_orig_min = float(xyz_out[:, 2].min()) if xyz_out.shape[0] else float("inf")
    z_orig_max = float(xyz_out[:, 2].max()) if xyz_out.shape[0] else float("-inf")
    n_pts = int(xyz_def.shape[0])
    n_hit = int((H > 1e-9).sum())

    stats["z_orig_min"] = z_orig_min
    stats["z_orig_max"] = z_orig_max
    stats["t_parse_s"] = t1 - t0
    stats["t_transform_s"] = t2 - t1
    stats["t_write_s"] = t3 - t2
    stats["n_pts"] = n_pts
    stats["n_pts_in_footprint"] = n_hit
    stats["target_bounds"] = tgt_bounds
    stats["H_max"] = float(H.max()) if H.size else 0.0
    stats["H_mean_in_footprint"] = float(H[H > 1e-9].mean()) if n_hit else 0.0

    if verbose:
        print(
            f"[conform] {in_path.name} -> {out_path.name}: "
            f"{stats['lines_in']:,} -> {stats['lines_out']:,} lines "
            f"({stats['moves_in']:,} moves -> {stats['moves_out']:,}; "
            f"subdiv≈{stats['moves_out']/max(1,stats['moves_in']):.1f}x)"
        )
        print(
            f"[conform] target STL bounds: X[{tgt_bounds[0][0]:.1f},{tgt_bounds[1][0]:.1f}] "
            f"Y[{tgt_bounds[0][1]:.1f},{tgt_bounds[1][1]:.1f}] "
            f"Z[{tgt_bounds[0][2]:.2f},{tgt_bounds[1][2]:.2f}] mm"
        )
        print(
            f"[conform] H per gcode pt: max {stats['H_max']:.3f} mm, "
            f"{n_hit:,}/{n_pts:,} pts in footprint "
            f"(mean offset {stats['H_mean_in_footprint']:.3f} mm)"
        )
        print(
            f"[conform] planar Z: [{stats['z_min']:.3f}, {stats['z_max']:.3f}] mm "
            f"→ conformed Z: [{z_orig_min:.3f}, {z_orig_max:.3f}] mm"
        )
        print(
            f"[conform] timing: parse {stats['t_parse_s']:.2f}s "
            f"transform {stats['t_transform_s']:.2f}s ({n_pts:,} pts) "
            f"write {stats['t_write_s']:.2f}s"
        )
    return stats


def _transform_chunk_worker(
    direction: str,
    xyz_chunk: np.ndarray,
    depth_field: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    dz_per_layer: float,
    bed_z: float,
    bed_blend_height: float,
) -> np.ndarray:
    """Multiprocessing worker for backtransform_gcode_file's parallel path."""
    bt = BackTransform(
        depth_field=depth_field,
        origin=origin,
        pitch=pitch,
        dz_per_layer=dz_per_layer,
        bed_z=bed_z,
        bed_blend_height=bed_blend_height,
    )
    if direction == "forward":
        return bt.forward_points_batch(xyz_chunk)
    return bt.invert_points_batch(xyz_chunk)
