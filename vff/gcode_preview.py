"""G-code preview: parse a 3-axis G-code file into a line-segment polyline
so the VFF viewer can draw it. PrusaSlicer's own preview can't display
non-planar layers — it expects clean horizontal Z stops. We need our
own renderer for the backtransformed output.

We only care about *extrusion* moves (G1 with positive E delta) for the
print path; travel moves (G0, G1 with E≤0) are kept separately and can
be toggled on/off. M82/M83 are tracked so absolute and relative E both
work. G92 updates state.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_TOKEN_RE = re.compile(r"([A-Z])\s*(-?(?:\d+\.\d*|\.\d+|\d+))")


def _parse_xyzef(tokens: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    rest = " ".join(tokens)
    for letter, val in _TOKEN_RE.findall(rest):
        if letter in "XYZEF":
            out[letter] = float(val)
    return out


def parse_gcode(path: str | Path) -> dict:
    """Parse a G-code file into extrusion and travel polylines.

    Returns a dict:
        extrusion_points : (Ne, 3) float64 — pairs of (start, end) for Ne/2
                           extrusion segments. Concatenated as [s0, e0, s1, e1, ...].
        extrusion_lines  : (Ne_segments * 3,) int64 — VTK line connectivity
                           [2, i0, i1, 2, i2, i3, ...]
        extrusion_step   : (Ne_segments,) float32 — Z of segment midpoint
                           (used for colouring by height).
        extrusion_rate   : (Ne_segments,) float32 — E_delta / segment_length.
        travel_points    : same shape but for non-extrusion XYZ moves.
        travel_lines     : same.
        n_extrusion_moves: int
        n_travel_moves   : int
    """
    in_path = Path(path)
    cur_x = cur_y = cur_z = 0.0
    cur_e = 0.0
    e_relative = True

    ext_pts: list[tuple[float, float, float]] = []  # alternating start, end, ...
    ext_step: list[float] = []
    ext_rate: list[float] = []
    trv_pts: list[tuple[float, float, float]] = []

    with in_path.open("r", encoding="utf-8", errors="replace") as fi:
        for raw in fi:
            line = raw.rstrip("\r\n").lstrip()
            if not line or line.startswith(";"):
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

            if cmd in ("G0", "G1", "G00", "G01"):
                params = _parse_xyzef(tokens[1:])
                new_x = params.get("X", cur_x)
                new_y = params.get("Y", cur_y)
                new_z = params.get("Z", cur_z)
                e_val = params.get("E", None)

                # Determine extrusion delta and the move's classification.
                if e_val is not None:
                    if e_relative:
                        de = e_val
                    else:
                        de = e_val - cur_e
                        cur_e = e_val
                else:
                    de = 0.0

                moved = (new_x != cur_x) or (new_y != cur_y) or (new_z != cur_z)
                if moved:
                    if de > 1e-9 and cmd in ("G1", "G01"):
                        ext_pts.append((cur_x, cur_y, cur_z))
                        ext_pts.append((new_x, new_y, new_z))
                        seg_len = float(np.hypot(
                            np.hypot(new_x - cur_x, new_y - cur_y), new_z - cur_z
                        ))
                        ext_step.append(0.5 * (cur_z + new_z))
                        ext_rate.append(de / max(seg_len, 1e-9))
                    else:
                        trv_pts.append((cur_x, cur_y, cur_z))
                        trv_pts.append((new_x, new_y, new_z))

                cur_x, cur_y, cur_z = new_x, new_y, new_z
                continue

            if cmd == "G92":
                params = _parse_xyzef(tokens[1:])
                if "X" in params: cur_x = params["X"]
                if "Y" in params: cur_y = params["Y"]
                if "Z" in params: cur_z = params["Z"]
                if "E" in params: cur_e = params["E"]
                continue

    def _build_polydata_arrays(pts: list[tuple[float, float, float]]):
        if not pts:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
            )
        arr = np.asarray(pts, dtype=np.float64)
        n_segments = arr.shape[0] // 2
        lines = np.empty(3 * n_segments, dtype=np.int64)
        lines[0::3] = 2
        lines[1::3] = np.arange(n_segments, dtype=np.int64) * 2
        lines[2::3] = np.arange(n_segments, dtype=np.int64) * 2 + 1
        return arr, lines

    ext_arr, ext_lines = _build_polydata_arrays(ext_pts)
    trv_arr, trv_lines = _build_polydata_arrays(trv_pts)

    return {
        "extrusion_points": ext_arr,
        "extrusion_lines": ext_lines,
        "extrusion_step": np.asarray(ext_step, dtype=np.float32),
        "extrusion_rate": np.asarray(ext_rate, dtype=np.float32),
        "travel_points": trv_arr,
        "travel_lines": trv_lines,
        "n_extrusion_moves": len(ext_step),
        "n_travel_moves": trv_arr.shape[0] // 2,
    }
