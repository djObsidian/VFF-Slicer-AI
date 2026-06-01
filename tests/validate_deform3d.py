"""Validate the full-3D straightening map vs the Z-only deform, on the real
propeller. Not a pass/fail unit test — a diagnostic report.

Run: .venv/Scripts/python.exe tests/validate_deform3d.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# This script runs outside the vff CLI, so apply the same UTF-8 console guard
# the package's __main__ does (Windows cp1251 can't encode ³/∘/→).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from vff.build_volume import BuildVolume
from vff.deform import deform_mesh
from vff.deform3d import deform_mesh_3d, solve_deformation_map
from vff.growth import compute_growth
from vff.mesh_io import load_and_place
from vff.voxelize import voxelize_solid


def edge_length_stats(orig, deformed):
    """Coefficient of variation of per-edge length ratio (deformed/original).
    Lower = closer to an isometry (less in-plane distortion). 0 = uniform."""
    e = orig.edges_unique
    o = np.linalg.norm(orig.vertices[e[:, 0]] - orig.vertices[e[:, 1]], axis=1)
    d = np.linalg.norm(deformed.vertices[e[:, 0]] - deformed.vertices[e[:, 1]], axis=1)
    ok = o > 1e-9
    ratio = d[ok] / o[ok]
    return float(np.mean(ratio)), float(np.std(ratio)), float(np.std(ratio) / np.mean(ratio))


def main() -> int:
    stl = Path(__file__).resolve().parent.parent / "propeller.stl"
    pitch = 1.0
    m = load_and_place(str(stl), BuildVolume.of(250, 250, 250))
    z0, z1 = float(m.bounds[0, 2]), float(m.bounds[1, 2])
    print(f"propeller: {len(m.faces):,} tris, Z [{z0:.2f}, {z1:.2f}] (extent {z1 - z0:.2f} mm), "
          f"vol {m.volume:.0f} mm³")

    vg = voxelize_solid(m, pitch=pitch)
    gr = compute_growth(vg, max_tilt_deg=30.0)
    print(f"voxels: {vg.shape} filled {vg.filled_count:,}, growth n_steps {gr.n_steps}")

    # --- Z-only baseline (dz auto so Z-extent ~ matches, for a fair scale) ---
    dm_z = deform_mesh(m, gr, dz_per_layer=None, depth_method="vectors")  # dz=pitch
    mz, sz, cvz = edge_length_stats(m, dm_z)
    print("\n[Z-only deform_mesh]")
    print(f"  deformed Z extent : {dm_z.bounds[1,2]-dm_z.bounds[0,2]:.2f} mm")
    print(f"  volume ratio      : {dm_z.volume / m.volume:.3f}")
    print(f"  edge ratio        : mean {mz:.3f}, CoV {cvz:.4f}  (in-plane distortion)")

    # --- Full 3D ---
    t0 = time.perf_counter()
    dmap = solve_deformation_map(gr)
    t1 = time.perf_counter()
    dm3 = deform_mesh_3d(m, dmap)
    m3, s3, cv3 = edge_length_stats(m, dm3)
    print(f"\n[Full-3D deform3d]  (solve {t1-t0:.2f}s)")
    print(f"  deformed Z extent : {dm3.bounds[1,2]-dm3.bounds[0,2]:.2f} mm")
    print(f"  deformed XY extent: X {dm3.bounds[1,0]-dm3.bounds[0,0]:.1f}  Y {dm3.bounds[1,1]-dm3.bounds[0,1]:.1f} mm "
          f"(orig X {m.bounds[1,0]-m.bounds[0,0]:.1f} Y {m.bounds[1,1]-m.bounds[0,1]:.1f})")
    print(f"  volume ratio      : {dm3.volume / m.volume:.3f}")
    print(f"  edge ratio        : mean {m3:.3f}, CoV {cv3:.4f}  (in-plane distortion)")
    print(f"  --> distortion CoV {cv3:.4f} vs Z-only {cvz:.4f}  "
          f"({'BETTER' if cv3 < cvz else 'WORSE'}: {cvz/max(cv3,1e-9):.2f}x)")

    # --- bed pinned? deformed bed vertices should match original ---
    bed_v = m.vertices[:, 2] < z0 + 0.5 * pitch
    if bed_v.any():
        disp = np.linalg.norm(dm3.vertices[bed_v] - m.vertices[bed_v], axis=1)
        print(f"\n[bed pinning] {bed_v.sum()} bed verts: max disp {disp.max():.3f} mm, "
              f"mean {disp.mean():.3f} mm (want ~0)")

    # --- Newton inverse: forward∘inverse ≈ identity on random interior pts ---
    rng = np.random.default_rng(0)
    ox, oy, oz = dmap.origin
    nx, ny, nz = dmap.phi.shape[:3]
    pts = np.column_stack([
        rng.uniform(ox + 2, ox + nx - 2, 5000),
        rng.uniform(oy + 2, oy + ny - 2, 5000),
        rng.uniform(oz + 2, oz + nz - 2, 5000),
    ])
    q = dmap.forward_points(pts)
    t2 = time.perf_counter()
    back, conv = dmap.inverse_points(q)
    t3 = time.perf_counter()
    err = np.linalg.norm(back - pts, axis=1)
    print(f"\n[Newton inverse] {len(pts):,} pts in {t3-t2:.2f}s, converged {100*conv.mean():.1f}%")
    print(f"  forward∘inverse error: median {np.median(err):.4f}, p95 {np.percentile(err,95):.4f}, "
          f"max {err.max():.4f} mm")
    print(f"  (on converged pts: median {np.median(err[conv]):.5f}, "
          f"p95 {np.percentile(err[conv],95):.5f} mm)" if conv.any() else "  (none converged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
