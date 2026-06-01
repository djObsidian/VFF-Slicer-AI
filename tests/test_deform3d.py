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
from vff.deform3d import _face_nonaffinity, deform_mesh_3d, solve_deformation_map
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


def test_extrusion_comp_matches_volume_ratio():
    """Extrusion compensation = 1/det(JΦ). Mean over the model must equal the
    inverse volume ratio (material conservation); a flat box stays ~1."""
    box = trimesh.creation.box(extents=(40, 40, 20))
    box.apply_translation([125, 125, 10])
    grb = compute_growth(voxelize_solid(box, pitch=1.0), max_tilt_deg=30.0)
    dmb = solve_deformation_map(grb)
    matb = grb.step >= 0
    Pb = dmb.origin + (np.argwhere(matb) + 0.5) * dmb.pitch
    detb = dmb.jacobian_det(Pb)
    assert np.allclose(detb, 1.0, atol=1e-3), f"box det should be ~1, got [{detb.min():.3f},{detb.max():.3f}]"

    stl = Path(__file__).resolve().parent.parent / "propeller.stl"
    if not stl.exists():
        print("  PASS extrusion comp (box det~1; propeller skipped — no STL)")
        return
    m = load_and_place(str(stl), BuildVolume.of(250, 250, 250))
    gr = compute_growth(voxelize_solid(m, pitch=1.0), max_tilt_deg=30.0)
    dmap = solve_deformation_map(gr)
    dm = deform_mesh_3d(m, dmap)
    mat = gr.step >= 0
    P = dmap.origin + (np.argwhere(mat) + 0.5) * dmap.pitch
    comp_mean = float(np.mean(1.0 / dmap.jacobian_det(P)))
    inv_vol_ratio = float(m.volume / dm.volume)
    assert abs(comp_mean - inv_vol_ratio) < 0.02, (
        f"mean comp {comp_mean:.3f} should match 1/vol-ratio {inv_vol_ratio:.3f}")
    print(f"  PASS extrusion comp (box det~1; propeller mean comp {comp_mean:.3f} "
          f"== 1/vol-ratio {inv_vol_ratio:.3f})")


def test_subdivision_refines_coarse_faces_watertight():
    """Coarse flat faces deform with large per-face error; --subdivide-error
    refines them and the result stays watertight (uniform → no T-cracks)."""
    stl = Path(__file__).resolve().parent.parent / "propeller_fixed_flat.stl"
    if not stl.exists():
        print("  SKIP subdivision (propeller_fixed_flat.stl not found)")
        return
    m = load_and_place(str(stl), BuildVolume.of(250, 250, 250))
    gr = compute_growth(voxelize_solid(m, pitch=1.0), max_tilt_deg=30.0)
    dmap = solve_deformation_map(gr)
    err0 = float(_face_nonaffinity(m, dmap).max())
    assert err0 > 0.1, f"expected coarse faces with >0.1mm error, got {err0:.3f}"

    coarse = deform_mesh_3d(m, dmap)                                  # no subdiv
    fine = deform_mesh_3d(m, dmap, subdivide_max_error=0.1)           # auto subdiv
    assert len(fine.faces) > len(coarse.faces), "subdivision must add faces"
    assert fine.is_watertight, "uniform subdivision must stay watertight (no cracks)"
    print(f"  PASS subdivision ({err0:.2f}mm coarse err; {len(coarse.faces):,}->"
          f"{len(fine.faces):,} faces, watertight {fine.is_watertight})")


def test_vertical_extrusion_comp_reduces_e_in_bulk():
    """3-axis 'vertical' comp = layer-gap ratio ∂orig_z/∂def_z. A flat box (no
    deformation) gives ratio 1 (no comp); on the propeller the layers compress
    so the mean is < 1 (reduces E), opposite to the volume comp's > 1 — that
    sign flip is the over-extrusion fix."""
    box = trimesh.creation.box(extents=(40, 40, 20))
    box.apply_translation([125, 125, 10])
    grb = compute_growth(voxelize_solid(box, pitch=1.0), max_tilt_deg=30.0)
    db = solve_deformation_map(grb)
    Pb = db.origin + (np.argwhere(grb.step >= 0) + 0.5) * db.pitch
    assert np.allclose(db.layer_gap_ratio(Pb), 1.0, atol=1e-2), "box must give layer ratio ~1"

    stl = Path(__file__).resolve().parent.parent / "propeller_fixed_flat.stl"
    if not stl.exists():
        print("  PASS vertical comp (box ratio ~1; propeller skipped)")
        return
    m = load_and_place(str(stl), BuildVolume.of(250, 250, 250))
    gr = compute_growth(voxelize_solid(m, pitch=1.0), max_tilt_deg=30.0)
    dmap = solve_deformation_map(gr)
    P = dmap.origin + (np.argwhere(gr.step >= 0) + 0.5) * dmap.pitch
    vert = float(np.mean(dmap.layer_gap_ratio(P)))
    vol = float(np.mean(1.0 / dmap.jacobian_det(P)))
    assert vert < 1.0, f"vertical comp should reduce E in the bulk, mean {vert:.3f}"
    assert vert < vol, f"vertical {vert:.3f} must be below volume {vol:.3f} (opposite tweak)"
    print(f"  PASS vertical comp (box ~1; propeller mean x{vert:.3f} < volume x{vol:.3f})")


def _bore_ceiling_dome(m, dmap):
    cx, cy = 125.0, 125.0
    rs = np.linspace(0, 6, 13)
    pts = np.column_stack([cx + rs, np.full(13, cy), np.full(13, 9.0)])
    ins = m.contains(pts)
    if ins.sum() < 3:
        return None
    pz = dmap.forward_points(pts)[:, 2]
    return float(pz[ins][0] - pz[ins][-1])


def test_default_smoothing_keeps_a_bore_dome():
    """The deformation must dome the bore ceiling (the depth field dips there).
    The displacement-smoothing default (2.0) softens it to ~0.7 mm — fine, the
    dome prints once --subdivide-error refines the coarse ceiling triangle. Guard
    against a regression that crushes the field dome to ~0 (over-smoothing)."""
    stl = Path(__file__).resolve().parent.parent / "propeller_fixed_flat.stl"
    if not stl.exists():
        print("  SKIP bore dome (propeller_fixed_flat.stl not found)")
        return
    m = load_and_place(str(stl), BuildVolume.of(250, 250, 250))
    gr = compute_growth(voxelize_solid(m, pitch=1.0), max_tilt_deg=30.0)
    dome = _bore_ceiling_dome(m, solve_deformation_map(gr))  # default sigma
    assert dome is not None and dome > 0.5, (
        f"deformation must keep a bore dome (>0.5 mm), got {dome:.2f} mm "
        "— displacement_smooth_sigma is probably too high")
    print(f"  PASS deformation keeps a bore dome ({dome:.2f} mm at default sigma)")


def test_geodesic_growth_source_domes_more_box_identity():
    """'geodesic' growth source domes a bore ceiling more than 'bfs' (captures
    the detour around the hole) while keeping a flat box at identity."""
    box = trimesh.creation.box(extents=(40, 40, 20))
    box.apply_translation([125, 125, 10])
    grb = compute_growth(voxelize_solid(box, pitch=1.0), max_tilt_deg=30.0)
    db = solve_deformation_map(grb, growth_source="geodesic", max_tilt_deg=30.0)
    disp = float(np.linalg.norm(db.forward_points(box.vertices) - box.vertices, axis=1).max())
    assert disp < 1e-2, f"geodesic box must stay identity, max disp {disp:.4f}"

    stl = Path(__file__).resolve().parent.parent / "propeller_fixed_flat.stl"
    if not stl.exists():
        print(f"  PASS geodesic (box identity {disp:.2e}; propeller skipped)")
        return
    m = load_and_place(str(stl), BuildVolume.of(250, 250, 250))
    gr = compute_growth(voxelize_solid(m, pitch=1.0), max_tilt_deg=30.0)
    dome_bfs = _bore_ceiling_dome(m, solve_deformation_map(gr, growth_source="bfs"))
    dome_geo = _bore_ceiling_dome(
        m, solve_deformation_map(gr, growth_source="geodesic", max_tilt_deg=30.0))
    assert dome_geo is not None and dome_bfs is not None
    assert dome_geo > dome_bfs + 0.3, f"geodesic dome {dome_geo:.2f} should exceed bfs {dome_bfs:.2f}"
    print(f"  PASS geodesic (box identity {disp:.2e}; bore dome bfs {dome_bfs:.2f} -> "
          f"geodesic {dome_geo:.2f} mm)")


def main() -> int:
    tests = [
        test_flat_box_maps_to_identity,
        test_propeller_less_distortion_than_zonly,
        test_inverse_roundtrip_converged_is_exact,
        test_extrusion_comp_matches_volume_ratio,
        test_vertical_extrusion_comp_reduces_e_in_bulk,
        test_subdivision_refines_coarse_faces_watertight,
        test_default_smoothing_keeps_a_bore_dome,
        test_geodesic_growth_source_domes_more_box_identity,
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
