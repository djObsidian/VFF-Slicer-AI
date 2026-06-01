"""Validate an inverse-transformed G-code against the original target mesh.

The inverse backtransform should map the planar-sliced G-code of the
3D-deformed mesh back INTO the shape of the original (un-deformed) target.
This script measures that: it places the target STL at the inverse gcode's
own XY centre and Z_min=0, then checks how many extrusion points land inside
the target, plus bbox agreement and any below-bed extrusion.

Usage:
    .venv/Scripts/python.exe tests/validate_inverse_gcode.py \
        propeller_inverse_3d.gcode propeller_fixed_flat.stl
    (defaults to those two if omitted)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from vff.gcode_preview import parse_gcode


def main(argv: list[str]) -> int:
    gcode = Path(argv[0]) if len(argv) > 0 else ROOT / "propeller_inverse_3d.gcode"
    target = Path(argv[1]) if len(argv) > 1 else ROOT / "propeller_fixed_flat.stl"
    if not gcode.exists():
        print(f"inverse gcode not found: {gcode} (slice the 3D-deformed mesh first)")
        return 1
    if not target.exists():
        print(f"target STL not found: {target}")
        return 1

    d = parse_gcode(gcode)
    ep = d["extrusion_points"]
    if ep.shape[0] == 0:
        print("no extrusion moves in gcode")
        return 1

    # Place target at the inverse gcode's own XY centre, Z_min=0 (the frame the
    # inverse output lives in).
    gc = (0.5 * (ep[:, 0].min() + ep[:, 0].max()), 0.5 * (ep[:, 1].min() + ep[:, 1].max()))
    m = trimesh.load(str(target), force="mesh")
    m.apply_translation([0, 0, -m.bounds[0, 2]])
    c = 0.5 * (m.bounds[0, :2] + m.bounds[1, :2])
    m.apply_translation([gc[0] - c[0], gc[1] - c[1], 0])

    print(f"gcode    : {gcode.name}  ({ep.shape[0]:,} extrusion pts)")
    print(f"target   : {target.name}  Z[{m.bounds[0,2]:.2f},{m.bounds[1,2]:.2f}]")
    print(f"gcode bbox : X[{ep[:,0].min():.1f},{ep[:,0].max():.1f}] "
          f"Y[{ep[:,1].min():.1f},{ep[:,1].max():.1f}] Z[{ep[:,2].min():.2f},{ep[:,2].max():.2f}]")
    print(f"target bbox: X[{m.bounds[0,0]:.1f},{m.bounds[1,0]:.1f}] "
          f"Y[{m.bounds[0,1]:.1f},{m.bounds[1,1]:.1f}] Z[{m.bounds[0,2]:.2f},{m.bounds[1,2]:.2f}]")

    below = int((ep[:, 2] < -1e-6).sum())
    print(f"extrusion below bed (Z<0): {below:,} ({100*below/ep.shape[0]:.2f}%)")

    rng = np.random.default_rng(0)
    samp = ep[rng.choice(ep.shape[0], size=min(60000, ep.shape[0]), replace=False)]
    inside = m.contains(samp)
    print(f"extrusion INSIDE target  : {100*inside.mean():.1f}% "
          f"(rest are perimeter walls sitting on the surface)")

    ok = inside.mean() > 0.9 and below < 0.01 * ep.shape[0]
    print("\nRESULT:", "PASS — inverse reconstructs the target shape" if ok
          else "CHECK — low containment or below-bed extrusion")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
