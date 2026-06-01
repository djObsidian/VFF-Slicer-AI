"""Headless regression tests for the G-code transform layer.

Run: .venv/Scripts/python.exe -m tests.test_gcode_transform
(or  python tests/test_gcode_transform.py)

Plain asserts, no pytest dependency. Exit code 0 = all pass.
Covers the E-distribution correctness that the absolute-E subdivision
bug broke, plus a forward/inverse round-trip on the real propeller mesh.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

# Allow running as a loose script (python tests/test_gcode_transform.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vff.backtransform import (  # noqa: E402
    BackTransform,
    backtransform_gcode_file,
)

_E_RE = re.compile(r"\bE(-?(?:\d+\.\d*|\.\d+|\d+))")
_MOVE_RE = re.compile(r"^(G0|G1)\b")


def _identity_bt() -> BackTransform:
    """A BackTransform whose forward map leaves Z essentially unchanged
    (depth[i,j,k] == k, dz == pitch, so depth*dz == world z at cell centre).
    Geometry is irrelevant to E-distribution; this just keeps numbers sane."""
    nz = 40
    nx = ny = 4
    k = np.arange(nz, dtype=np.float32)
    depth = np.broadcast_to(k[None, None, :], (nx, ny, nz)).astype(np.float32)
    return BackTransform(
        depth_field=depth,
        origin=np.zeros(3),
        pitch=1.0,
        dz_per_layer=1.0,
        bed_z=0.0,
        bed_blend_height=0.0,  # no bed blend → pure map, simpler to reason about
    )


def _run(gcode_text: str, direction: str = "forward", subdiv_mm: float = 1.0,
         z_slowdown: float = 1.0, max_z_speed: float = 0.0) -> list[str]:
    bt = _identity_bt()
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.gcode"
        op = Path(td) / "out.gcode"
        ip.write_text(gcode_text, encoding="utf-8")
        backtransform_gcode_file(
            ip, op, bt, subdiv_mm=subdiv_mm, n_jobs=1,
            verbose=False, direction=direction, z_slowdown=z_slowdown,
            max_z_speed=max_z_speed,
        )
        return op.read_text(encoding="utf-8").splitlines()


def _move_e_values(lines: list[str]) -> list[float]:
    """E values on G0/G1 move lines, in order (skips non-move lines)."""
    out = []
    for ln in lines:
        if not _MOVE_RE.match(ln.strip()):
            continue
        m = _E_RE.search(ln)
        if m:
            out.append(float(m.group(1)))
    return out


def test_relative_e_subdivision_conserves_and_spreads():
    """M83: a 5mm extrusion of E=10, subdiv 1mm → 5 pieces of E=2 each."""
    gcode = "M83\nG1 X0 Y0 Z1 F1800\nG1 X5 Y0 Z1 E10\n"
    e_vals = _move_e_values(_run(gcode))
    assert len(e_vals) == 5, f"expected 5 extruding pieces, got {len(e_vals)}: {e_vals}"
    assert abs(sum(e_vals) - 10.0) < 1e-4, f"relative E not conserved: sum={sum(e_vals)}"
    for v in e_vals:
        assert abs(v - 2.0) < 1e-4, f"relative E not evenly spread: {e_vals}"
    print("  PASS relative-E subdivision spreads evenly, conserves total")


def test_absolute_e_subdivision_spreads_monotonically():
    """M82: absolute E 0→10 over a 5mm move, subdiv 1mm → 5 pieces.

    Correct: each piece carries a strictly-increasing absolute E
    (2,4,6,8,10) so the printer deposits filament along the whole move.
    The bug emitted E only on the final piece (0,0,0,0,10) → a blob.
    """
    gcode = "M82\nG92 E0\nG1 X0 Y0 Z1 F1800\nG1 X5 Y0 Z1 E10\n"
    e_vals = _move_e_values(_run(gcode))
    assert len(e_vals) == 5, (
        f"absolute E should appear on every piece, got {len(e_vals)}: {e_vals} "
        "(0/0/0/0/10 = the dump-in-last-segment bug)"
    )
    assert abs(e_vals[-1] - 10.0) < 1e-4, f"final absolute E must be 10, got {e_vals[-1]}"
    deltas = np.diff([0.0] + e_vals)
    assert (deltas > 0).all(), f"absolute E must increase every piece: {e_vals}"
    assert np.allclose(deltas, 2.0, atol=1e-3), f"E increments should be ~2mm each: {deltas}"
    print("  PASS absolute-E subdivision spreads monotonically, conserves total")


def test_absolute_e_with_running_start():
    """M82 with a non-zero starting E (no G92 reset): 4→14 over 4mm, subdiv 1.

    Pieces must be 6.5, 9, 11.5, 14 (start 4 + delta 10 spread over 4)."""
    gcode = "M82\nG92 E4\nG1 X0 Y0 Z2 F1800\nG1 X4 Y0 Z2 E14\n"
    e_vals = _move_e_values(_run(gcode))
    assert len(e_vals) == 4, f"expected 4 pieces, got {e_vals}"
    expected = [6.5, 9.0, 11.5, 14.0]
    assert np.allclose(e_vals, expected, atol=1e-3), f"got {e_vals}, want {expected}"
    print("  PASS absolute-E honours running start position")


def test_z_slowdown_scales_steep_move_feedrate():
    """--z-slowdown leaves flat moves at full F and halves (×0.5) a 45° move
    (slope 0.71 > the 0.5 reference). Default (1.0) leaves everything alone."""
    # X10 @ Z5 = flat extrusion; X11 @ Z6 = a 45° climb.
    gcode = "M83\nG1 X0 Y0 Z5 F6000\nG1 X10 Y0 Z5 E1\nG1 X11 Y0 Z6 E0.2\n"

    def f_at(lines, xtag):
        f = None
        for ln in lines:
            if not _MOVE_RE.match(ln.strip()):
                continue
            m = _E_RE.sub("", ln)  # avoid matching E
            mf = re.search(r"\bF(-?[0-9.]+)", ln)
            if mf:
                f = float(mf.group(1))
            if xtag in ln:
                return f
        return None

    base = _run(gcode, direction="inverse", subdiv_mm=100, z_slowdown=1.0)
    slow = _run(gcode, direction="inverse", subdiv_mm=100, z_slowdown=0.5)
    assert f_at(base, "X11.000") == 6000, "default must not touch F"
    assert abs(f_at(slow, "X10.000") - 6000) < 1, "flat move must stay full speed"
    assert abs(f_at(slow, "X11.000") - 3000) < 1, "45° move must be halved by z_slowdown=0.5"
    print("  PASS z-slowdown (flat F 6000 kept; 45° move 6000 -> 3000)")


def test_max_z_speed_hard_caps_z_velocity():
    """--max-z-speed bounds the Z-velocity component F·|dz|/L. A 45° move at
    F6000 climbs Z at 6000·1/√2 ≈ 4243 mm/min ≈ 70.7 mm/s; capping at 15 mm/s
    recomputes F = 15·60·√2 ≈ 1273 mm/min. Flat moves stay full speed; 0 = off."""
    import math
    gcode = "M83\nG1 X0 Y0 Z5 F6000\nG1 X10 Y0 Z5 E1\nG1 X11 Y0 Z6 E0.2\n"

    def f_at(lines, xtag):
        f = None
        for ln in lines:
            if not _MOVE_RE.match(ln.strip()):
                continue
            mf = re.search(r"\bF(-?[0-9.]+)", ln)
            if mf:
                f = float(mf.group(1))
            if xtag in ln:
                return f
        return None

    off = _run(gcode, direction="inverse", subdiv_mm=100, max_z_speed=0.0)
    cap = _run(gcode, direction="inverse", subdiv_mm=100, max_z_speed=15.0)
    expect = 15.0 * 60.0 * math.sqrt(2.0)        # ≈ 1272.79 mm/min
    assert f_at(off, "X11.000") == 6000, "off (0) must not touch F"
    assert abs(f_at(cap, "X10.000") - 6000) < 1, "flat move must stay full speed"
    assert abs(f_at(cap, "X11.000") - expect) < 1, (
        f"45° move Z-vel must be capped: want F≈{expect:.0f}, got {f_at(cap, 'X11.000')}")
    print(f"  PASS max-z-speed (flat F 6000 kept; 45° move capped 6000 -> {expect:.0f})")


def test_overhang_cooling_ramps_fan_by_severity():
    """Inverse cooling ramps the fan LINEARLY between cool_fan_min/max by the
    overhang degree (like a slicer's per-overlap fan curve): degree 0.5 → mid,
    1.0 → max, and the slicer's own fan is restored after the part is supported
    for `cool_linger` moves. Stub 3D BackTransform (identity inverse, a Z-band
    degree) keeps it deterministic and geometry-free."""
    class _StubBT:
        is_3d = True
        mesh = object()  # non-None: passes the gate
        def invert_points_batch(self, xyz):
            return np.asarray(xyz, dtype=np.float64).copy()  # identity
        def overhang_degree(self, xyz, probe=0.8, min_z=0.6):
            z = np.asarray(xyz, dtype=np.float64)[:, 2]
            d = np.zeros(len(z))
            d[(z >= 5) & (z < 8)] = 0.5   # partial overhang
            d[z >= 8] = 1.0               # full bridge
            return d

    gcode = (
        "M83\nM106 S80\n"
        "G1 X0 Y0 Z1 F1800\n"
        "G1 X1 Y0 Z1 E1\n"   # supported
        "G1 X2 Y0 Z6 E1\n"   # degree 0.5 → 100 + 0.5*(200-100) = 150
        "G1 X3 Y0 Z9 E1\n"   # degree 1.0 → 200
        "G1 X4 Y0 Z1 E1\n"   # supported (linger 1)
        "G1 X5 Y0 Z1 E1\n"   # 2
        "G1 X6 Y0 Z1 E1\n"   # 3
        "G1 X7 Y0 Z1 E1\n"   # 4 (>3) → restore S80
    )
    with tempfile.TemporaryDirectory() as td:
        ip = Path(td) / "in.gcode"
        op = Path(td) / "out.gcode"
        ip.write_text(gcode, encoding="utf-8")
        stats = backtransform_gcode_file(
            ip, op, _StubBT(), subdiv_mm=100, n_jobs=1, verbose=False,
            direction="inverse", extrusion_comp=False,
            cool_overhangs=True, cool_fan_min=100, cool_fan_max=200,
        )
        lines = op.read_text(encoding="utf-8").splitlines()

    def idx(prefix):
        return [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    mid, full = idx("M106 S150"), idx("M106 S200")
    assert len(mid) == 1 and len(full) == 1, f"want one S150 + one S200: {lines}"
    assert mid[0] < full[0], "lighter overhang (S150) must come before the bridge (S200)"
    assert stats["n_cool_moves"] == 2 and stats["n_cool_boosts"] == 2, (
        f"want 2 overhang moves / 2 fan changes, got "
        f"{stats['n_cool_moves']}/{stats['n_cool_boosts']}")
    restore = [i for i in idx("M106 S80") if i > full[0]]
    assert restore, f"slicer fan S80 not restored after the overhang: {lines}"
    print("  PASS overhang cooling ramp (S150 @0.5 → S200 @1.0; S80 restored)")


def test_propeller_forward_inverse_roundtrip():
    """forward then inverse on the real mesh should recover XYZ within a
    fraction of the voxel pitch (interpolation error only)."""
    stl = Path(__file__).resolve().parent.parent / "propeller_fixed_flat.stl"
    if not stl.exists():
        print("  SKIP roundtrip (propeller_fixed_flat.stl not found)")
        return
    bt = BackTransform.from_mesh(
        str(stl), volume_side=250.0, pitch=1.0, max_tilt_deg=30.0,
        smooth_sigma=2.0, depth_method="vectors", dz_per_layer=1.0,
    )
    rng = np.random.default_rng(0)
    # Sample points inside the depth-field's world bounds, above the bed blend.
    ox, oy, oz = bt.origin
    nx, ny, nz = bt.shape
    pts = np.column_stack([
        rng.uniform(ox + 2, ox + nx - 2, 2000),
        rng.uniform(oy + 2, oy + ny - 2, 2000),
        rng.uniform(oz + 3, oz + nz - 2, 2000),
    ])
    deformed = bt.forward_points_batch(pts)
    back = bt.invert_points_batch(deformed)
    dz = np.abs(back[:, 2] - pts[:, 2])
    # XY is preserved exactly by construction.
    assert np.allclose(back[:, :2], pts[:, :2]), "XY must be preserved"
    med, p95 = float(np.median(dz)), float(np.percentile(dz, 95))
    assert med < 0.05, f"median Z roundtrip error too high: {med:.4f} mm"
    assert p95 < 0.5, f"p95 Z roundtrip error too high: {p95:.4f} mm"
    print(f"  PASS forward/inverse roundtrip (Z err median {med:.4f} / p95 {p95:.4f} mm)")


def main() -> int:
    tests = [
        test_relative_e_subdivision_conserves_and_spreads,
        test_absolute_e_subdivision_spreads_monotonically,
        test_absolute_e_with_running_start,
        test_z_slowdown_scales_steep_move_feedrate,
        test_max_z_speed_hard_caps_z_velocity,
        test_overhang_cooling_ramps_fan_by_severity,
        test_propeller_forward_inverse_roundtrip,
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
