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

from vff.deform import harmonic_potential_from_bed  # noqa: E402
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


def main() -> int:
    tests = [
        test_box_potential_is_linear_height,
        test_max_principle_no_overshoot,
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
