"""Regression tests for the full-3D straightening map (vff/deform3d.py).

Run: .venv/Scripts/python.exe tests/test_deform3d.py
Plain asserts, exit 0 = all pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


def _edge_cov(orig, deformed):
    e = orig.edges_unique
    o = np.linalg.norm(orig.vertices[e[:, 0]] - orig.vertices[e[:, 1]], axis=1)
    d = np.linalg.norm(deformed.vertices[e[:, 0]] - deformed.vertices[e[:, 1]], axis=1)
    ok = o > 1e-9
    r = d[ok] / o[ok]
    return float(np.std(r) / np.mean(r))


def test_flat_box_maps_to_identity():
    """Growth is purely vertical in a flat box → R=I → Φ must be identity."""
    box = trimesh.creation.box(extents=(40, 40, 20))
    box.apply_translation([125, 125, 10])
    vg = voxelize_solid(box, pitch=1.0)
    gr = compute_growth(vg, max_tilt_deg=30.0)
    dmap = solve_deformation_map(gr)
    dm = deform_mesh_3d(box, dmap)
    disp = float(np.linalg.norm(dm.vertices - box.vertices, axis=1).max())
    assert disp < 1e-2, f"flat box must map to identity, max disp={disp:.4f} mm"
    assert abs(dm.volume / box.volume - 1.0) < 1e-3, "flat box volume must be preserved"
    print(f"  PASS flat box → identity (max disp {disp:.2e} mm)")


def test_propeller_less_distortion_than_zonly():
    stl = Path(__file__).resolve().parent.parent / "propeller.stl"
    if not stl.exists():
        print("  SKIP propeller distortion (propeller.stl not found)")
        return
    m = load_and_place(str(stl), BuildVolume.of(250, 250, 250))
    vg = voxelize_solid(m, pitch=1.0)
    gr = compute_growth(vg, max_tilt_deg=30.0)

    cov_z = _edge_cov(m, deform_mesh(m, gr, depth_method="vectors"))
    dmap = solve_deformation_map(gr)
    dm3 = deform_mesh_3d(m, dmap)
    cov_3 = _edge_cov(m, dm3)
    vol3 = dm3.volume / m.volume

    assert cov_3 < cov_z, f"3D distortion {cov_3:.4f} should be < Z-only {cov_z:.4f}"
    assert vol3 > 0.9, f"3D volume ratio {vol3:.3f} should be > 0.9 (Z-only ~0.71)"
    print(f"  PASS 3D less distortion (CoV {cov_3:.4f} < Z-only {cov_z:.4f}; vol {vol3:.3f})")


def test_inverse_roundtrip_converged_is_exact():
    stl = Path(__file__).resolve().parent.parent / "propeller.stl"
    if not stl.exists():
        print("  SKIP inverse roundtrip (propeller.stl not found)")
        return
    m = load_and_place(str(stl), BuildVolume.of(250, 250, 250))
    vg = voxelize_solid(m, pitch=1.0)
    gr = compute_growth(vg, max_tilt_deg=30.0)
    dmap = solve_deformation_map(gr)

    rng = np.random.default_rng(1)
    ox, oy, oz = dmap.origin
    nx, ny, nz = dmap.phi.shape[:3]
    pts = np.column_stack([
        rng.uniform(ox + 2, ox + nx - 2, 4000),
        rng.uniform(oy + 2, oy + ny - 2, 4000),
        rng.uniform(oz + 2, oz + nz - 2, 4000),
    ])
    q = dmap.forward_points(pts)
    back, conv = dmap.inverse_points(q)
    assert conv.mean() > 0.9, f"inverse should converge on >90% of points, got {100*conv.mean():.1f}%"
    err = np.linalg.norm(back[conv] - pts[conv], axis=1)
    assert np.percentile(err, 99) < 1e-2, f"converged roundtrip p99 err {np.percentile(err,99):.4f} too high"
    print(f"  PASS inverse roundtrip ({100*conv.mean():.1f}% converged, p99 err {np.percentile(err,99):.2e} mm)")


def main() -> int:
    tests = [
        test_flat_box_maps_to_identity,
        test_propeller_less_distortion_than_zonly,
        test_inverse_roundtrip_converged_is_exact,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
