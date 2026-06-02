"""Headless tests for the harmonic layer potential
(deform.harmonic_potential_from_bed).

Run: python tests/test_harmonic.py   (exit 0 = all pass; plain asserts, no pytest)

Guards the properties the harmonic route exists for:
  - a straight column gets phi == height-above-bed (the unique linear harmonic
    solution), so layers are exact horizontal planes;
  - the discrete maximum principle: phi never exceeds the top boundary value,
    i.e. no interior maximum => no closed shells;
  - the Dirichlet-ELIMINATION fix. A penalty solve (big diagonal) left the
    source-free interior stuck at ~0 because CG halts on the penalty-dominated
    residual before the boundary values reach the bulk; both checks below would
    have caught it (the box error would be ~the full height, and the interior
    would be ~0 everywhere but the top).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vff.deform import harmonic_potential_from_bed, smoothed_depth_field  # noqa: E402
from vff.growth import compute_growth  # noqa: E402
from vff.voxelize import voxelize_solid  # noqa: E402


def _box_growth(extents=(10, 10, 20), pitch=1.0, tilt=30.0):
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(-box.bounds[0])      # min corner at origin (on the bed)
    return compute_growth(voxelize_solid(box, pitch=pitch), max_tilt_deg=tilt)


def test_box_potential_is_linear_height():
    """Straight column => phi == height-above-bed (flat horizontal layers).
    Also catches the penalty-solve bug: there the interior collapses to ~0."""
    gr = _box_growth()
    inside = gr.step >= 0
    phi = harmonic_potential_from_bed(gr)
    has = inside.any(axis=(0, 1))
    kbed, ktop = int(np.argmax(has)), int(np.max(np.where(has)))

    worst = 0.0
    for k in range(kbed, ktop + 1):
        lay = inside[:, :, k]
        if lay.any():
            worst = max(worst, abs(float(phi[:, :, k][lay].mean()) - (k - kbed)))
    assert worst < 0.05, f"box phi must equal height; max mean error {worst:.3f}"

    bed = np.zeros_like(inside)
    bed[:, :, kbed] = inside[:, :, kbed]
    frac0 = float((inside & (phi < 0.5) & ~bed).sum()) / int(inside.sum())
    assert frac0 < 0.02, (
        f"interior collapsed to ~0 ({100 * frac0:.1f}%) — Dirichlet elimination "
        "regressed to a penalty solve")
    print(f"  PASS box phi == height (max err {worst:.3f}; interior-zero {100 * frac0:.1f}%)")


def test_max_principle_no_overshoot():
    """phi stays in [0, top-height]: a graph-harmonic function attains its
    extrema on the boundary, so any value above the top would be an interior
    maximum (the seed of a closed shell). None is allowed."""
    gr = _box_growth(extents=(8, 8, 16), pitch=1.0)
    inside = gr.step >= 0
    phi = harmonic_potential_from_bed(gr)
    has = inside.any(axis=(0, 1))
    kbed, ktop = int(np.argmax(has)), int(np.max(np.where(has)))
    hi = ktop - kbed
    vals = phi[inside]
    assert vals.min() >= -1e-4, f"phi dipped below the bed value: {vals.min():.3f}"
    assert vals.max() <= hi + 1e-4, (
        f"phi {vals.max():.3f} exceeds the top height {hi} — interior maximum present")
    print(f"  PASS max principle (phi in [0, {hi}]; no overshoot => no interior peak)")


def _interior_section_maxima(field, inside, jc):
    """Count interior 2D maxima in the XZ slice at j=jc (model pixels whose 4
    in-model neighbours are all strictly lower) — each is the centre of a closed
    iso-surface loop in the section."""
    M = inside[:, jc, :]
    interior = M.copy()
    for ax, sh in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        interior &= np.roll(M, sh, axis=ax)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    P = np.where(M, field[:, jc, :], np.nan)
    mx = interior.copy()
    for ax, sh in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        mx &= P > np.roll(P, sh, axis=ax) + 1e-4
    return int(mx.sum())


def test_section_field_has_no_false_closed_surfaces():
    """The XZ section viz smooths the layer field. With outside_mode='vertical'
    the Gaussian bleeds the low air-height (k − k_bed) into a wide overhang where
    the inside path-distance is much higher, FABRICATING an interior maximum — a
    closed iso-surface the deformation does not actually have. 'extend' (what
    build_growth_surfaces now uses) is continuous across the boundary and
    introduces none, while still showing any GENUINE interior extremum.

    Checked on the real mushroom/dumbbell (np_test1) — the artifact is geometry-
    and resolution-specific (a wide cap fed through a neck), so a toy box won't
    reproduce it. The STL is gitignored; skip cleanly when it is absent (same as
    the deform3d / propeller tests)."""
    stl = Path(__file__).resolve().parent.parent / "example_in" / "np_test1.stl"
    if not stl.exists():
        print("  PASS section no-false-closed-surfaces [SKIP: example_in/np_test1.stl not found]")
        return
    from vff.build_volume import BuildVolume
    from vff.mesh_io import load_and_place
    mesh = load_and_place(str(stl), BuildVolume.of(220, 220, 110))
    gr = compute_growth(voxelize_solid(mesh, pitch=0.2), max_tilt_deg=30.0)
    inside = gr.step >= 0
    ys = np.where(inside.any(axis=(0, 2)))[0]
    jc = int((ys.min() + ys.max()) // 2)
    n_vertical = _interior_section_maxima(
        smoothed_depth_field(gr, sigma=2.0, method="vectors", outside_mode="vertical"),
        inside, jc)
    n_extend = _interior_section_maxima(
        smoothed_depth_field(gr, sigma=2.0, method="vectors", outside_mode="extend"),
        inside, jc)
    assert n_vertical > 0, (
        "expected np_test1 to reproduce the 'vertical' artifact (so the test "
        f"actually bites); got {n_vertical} — geometry/pitch may have changed")
    assert n_extend == 0, f"'extend' must introduce no false closed surface, got {n_extend}"
    print(f"  PASS section no false closed surfaces (vertical={n_vertical} → extend={n_extend})")


def main() -> int:
    tests = [
        test_box_potential_is_linear_height,
        test_max_principle_no_overshoot,
        test_section_field_has_no_false_closed_surfaces,
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
    sys.exit(main())
