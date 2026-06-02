import argparse
import faulthandler
import sys
import traceback
from pathlib import Path

import numpy as np

# --- Surface every kind of failure to the console BEFORE we import VTK ---
# 1. Unbuffered, UTF-8 stdio so prints survive a hard crash AND don't blow up
#    on a non-UTF console (Windows cp1251/cp866 etc. can't encode the →/≈/√
#    we use in banners; errors='replace' degrades gracefully instead of
#    raising UnicodeEncodeError mid-pipeline).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# 2. Native crash traceback (SIGSEGV, abort, etc.) -> stderr.
faulthandler.enable(file=sys.stderr, all_threads=True)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    # SystemExit / KeyboardInterrupt are normal exit signals — defer to the
    # default hook (which just exits, no traceback).
    if issubclass(exc_type, (SystemExit, KeyboardInterrupt)):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    print("[vff] UNCAUGHT EXCEPTION:", file=sys.stderr, flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()


sys.excepthook = _excepthook


def _run_gcode_transform(args, stl_path: Path, x: float, y: float, z: float) -> int:
    """Depth-field G-code transform (forward/inverse). Headless: builds the
    depth field and rewrites the G-code without ever importing the Viewer /
    pyvista, so batch slicing works on machines with no display."""
    from .backtransform import backtransform_gcode_file, quick_gcode_xy_bounds

    out_path = args.gcode_out
    if not out_path:
        in_p = Path(args.gcode_in)
        suffix = ".nonplanar.gcode" if args.gcode_direction == "forward" else ".planar.gcode"
        out_path = str(in_p.with_suffix(suffix))

    # Quick-scan the gcode XY range so the deformation is aligned to the slicer's
    # model placement (PrusaSlicer / Cura don't use our --volume centre).
    xy_center = None
    align_info = ""
    if not args.no_gcode_align:
        xy_min, xy_max = quick_gcode_xy_bounds(args.gcode_in)
        if np.isfinite(xy_min).all() and np.isfinite(xy_max).all():
            xy_center = (0.5 * (xy_min[0] + xy_max[0]), 0.5 * (xy_min[1] + xy_max[1]))
            align_info = (
                f"  align→gcode  : XY range [{xy_min[0]:.1f}, {xy_max[0]:.1f}] x "
                f"[{xy_min[1]:.1f}, {xy_max[1]:.1f}], centre ({xy_center[0]:.1f}, {xy_center[1]:.1f})\n"
            )

    # ---- Full-3D path: BackTransform3D moves all axes; no dz/depth knobs. ----
    if args.deform_mode == "3d":
        from .deform3d import BackTransform3D
        print(
            f"G-code transform 3d ({args.gcode_direction}): {args.gcode_in} -> {out_path}\n"
            f"  STL          : {stl_path}\n"
            f"  volume       : {x:.0f}x{y:.0f}x{z:.0f} mm\n"
            f"  pitch        : {args.pitch} mm\n"
            f"  max-tilt     : {args.max_tilt} deg\n"
            f"  smooth-sigma : {args.smooth_sigma} (must match the export)\n"
            f"  depth-method : {args.depth_method} (must match the export)\n"
            f"  extrusion-comp: {(args.extrusion_comp_mode if args.extrusion_comp else 'off')}\n"
            f"  cool-overhangs: {('S'+str(args.cool_fan_min)+'..'+str(args.cool_fan_max)+' ramped, probe '+str(args.cool_probe)+' mm' + (', speed '+str(args.cool_speed)+' mm/s' if args.cool_speed and args.cool_speed > 0 else ', no speed cap') if (args.cool_overhangs and args.gcode_direction == 'inverse') else 'off')}\n"
            f"  max-z-speed  : {(str(args.max_z_speed)+' mm/s' if args.max_z_speed and args.max_z_speed > 0 else 'off')}\n"
            f"{align_info}"
            f"  subdiv-mm    : {args.subdiv_mm} mm",
            flush=True,
        )
        bt3 = BackTransform3D.from_mesh(
            str(stl_path), volume_side=max(x, y, z), pitch=args.pitch,
            max_tilt_deg=args.max_tilt, smooth_sigma=args.smooth_sigma,
            depth_method=args.depth_method, xy_center=xy_center,
        )
        backtransform_gcode_file(
            args.gcode_in, out_path, bt3,
            subdiv_mm=args.subdiv_mm, n_jobs=1, direction=args.gcode_direction,
            extrusion_comp=args.extrusion_comp, extrusion_comp_mode=args.extrusion_comp_mode,
            z_slowdown=args.z_slowdown, max_z_speed=args.max_z_speed,
            cool_overhangs=args.cool_overhangs,
            cool_fan_min=args.cool_fan_min, cool_fan_max=args.cool_fan_max,
            cool_speed=args.cool_speed,
            cool_probe=args.cool_probe, cool_min_z=args.cool_min_z,
        )
        return 0

    # ---- Z-only path (legacy) ----
    from .backtransform import BackTransform

    # The inverse must undo the SAME deformation that produced the sliced mesh.
    # --export bakes the mesh with dz = --dz-per-layer (default = pitch), so the
    # inverse has to use that exact dz. --dz-auto-fit is a forward-only
    # convenience (it picks dz from the gcode/STL extent) and will generally NOT
    # match the export's dz → a wrong inverse. Warn loudly.
    if args.gcode_direction == "inverse" and args.dz_auto_fit:
        print(
            "  WARNING: --dz-auto-fit with --gcode-direction inverse. The inverse "
            "must reuse the dz the deformed mesh was exported with (default = "
            "pitch). Auto-fit picks a different dz and will mis-map the layers. "
            "Pass the same --dz-per-layer you used for --export instead.",
            file=sys.stderr, flush=True,
        )

    # Auto-fit picks dz so the forward-deformed Z extent matches the STL's Z
    # extent: build the depth field once, take its max over model voxels,
    # dz = z_extent / depth_max.
    chosen_dz = args.dz_per_layer
    autofit_info = ""
    if args.dz_auto_fit:
        import trimesh

        from .build_volume import BuildVolume
        from .deform import smoothed_depth_field
        from .growth import compute_growth
        from .mesh_io import load_and_place
        from .voxelize import voxelize_solid

        _m = trimesh.load(str(stl_path), force="mesh")
        if xy_center is not None:
            _mxy = 0.5 * (_m.bounds[0, :2] + _m.bounds[1, :2])
            _m.apply_translation([
                xy_center[0] - float(_mxy[0]),
                xy_center[1] - float(_mxy[1]),
                -float(_m.bounds[0, 2]),
            ])
        else:
            _m = load_and_place(str(stl_path), BuildVolume.of(x, y, z))
        _vg = voxelize_solid(_m, pitch=args.pitch)
        _gr = compute_growth(_vg, max_tilt_deg=args.max_tilt)
        _f = smoothed_depth_field(
            _gr, sigma=args.smooth_sigma, method=args.depth_method, outside_mode="extend"
        )
        _depth_max = float(np.nanmax(_f[_gr.step >= 0])) if (_gr.step >= 0).any() else 1.0
        _z_extent = float(_m.bounds[1, 2] - _m.bounds[0, 2])
        if _depth_max > 1e-9:
            chosen_dz = _z_extent / _depth_max
        autofit_info = (
            f"  auto-fit     : z_extent={_z_extent:.3f}, depth_max={_depth_max:.3f}, "
            f"dz_per_layer={chosen_dz:.4f}\n"
        )

    print(
        f"G-code transform ({args.gcode_direction}): {args.gcode_in} -> {out_path}\n"
        f"  STL          : {stl_path}\n"
        f"  volume       : {x:.0f}x{y:.0f}x{z:.0f} mm\n"
        f"  pitch        : {args.pitch} mm\n"
        f"  max-tilt     : {args.max_tilt} deg\n"
        f"  smooth-sigma : {args.smooth_sigma}\n"
        f"  depth-method : {args.depth_method}\n"
        f"  dz_per_layer : {chosen_dz if chosen_dz is not None else 'pitch ('+str(args.pitch)+')'}\n"
        f"{align_info}"
        f"{autofit_info}"
        f"  subdiv-mm    : {args.subdiv_mm} mm",
        flush=True,
    )
    bt = BackTransform.from_mesh(
        str(stl_path),
        volume_side=max(x, y, z),
        pitch=args.pitch,
        max_tilt_deg=args.max_tilt,
        smooth_sigma=args.smooth_sigma,
        depth_method=args.depth_method,
        dz_per_layer=chosen_dz,
        xy_center=xy_center,
    )
    backtransform_gcode_file(
        args.gcode_in, out_path, bt,
        subdiv_mm=args.subdiv_mm, n_jobs=args.jobs,
        direction=args.gcode_direction, z_slowdown=args.z_slowdown,
        max_z_speed=args.max_z_speed,
    )
    return 0


def _run_3d_export(args, stl_path: Path, x: float, y: float, z: float) -> int:
    """Headless full-3D deformed-mesh export (no Viewer / pyvista)."""
    from .build_volume import BuildVolume
    from .deform3d import _face_nonaffinity, deform_mesh_3d, solve_deformation_map
    from .growth import compute_growth
    from .mesh_io import load_and_place
    from .voxelize import voxelize_solid

    mesh = load_and_place(str(stl_path), BuildVolume.of(x, y, z))
    print(
        f"Full-3D deform export: {stl_path} -> {args.export}\n"
        f"  volume {x:.0f}x{y:.0f}x{z:.0f} mm, pitch {args.pitch} mm, "
        f"max-tilt {args.max_tilt} deg, smooth-sigma {args.smooth_sigma}, "
        f"depth-method {args.depth_method}, "
        f"subdivide-error {args.subdivide_error} mm ({args.remesh} remesh)",
        flush=True,
    )
    vg = voxelize_solid(mesh, pitch=args.pitch)
    gr = compute_growth(vg, max_tilt_deg=args.max_tilt)
    dmap = solve_deformation_map(
        gr, displacement_smooth_sigma=args.smooth_sigma, max_tilt_deg=args.max_tilt,
        depth_method=args.depth_method,
    )
    base_err = float(_face_nonaffinity(mesh, dmap).max())
    dm = deform_mesh_3d(
        mesh, dmap, subdivide_max_error=args.subdivide_error, refine=args.remesh,
    )
    dm.export(args.export)
    vr = (dm.volume / mesh.volume) if mesh.volume > 0 else 0.0
    print(
        f"  saved: {len(mesh.faces):,} -> {len(dm.faces):,} faces, "
        f"Z [{dm.bounds[0, 2]:.2f}, {dm.bounds[1, 2]:.2f}] mm, volume ratio {vr:.3f}\n"
        f"  per-face deform error: {base_err:.3f} mm at input resolution",
        flush=True,
    )
    if args.subdivide_error <= 0 and base_err > 0.1:
        print(
            f"  NOTE: {base_err:.2f} mm of deformation error on coarse faces "
            "(flat regions stay flat) and refinement is off. Drop "
            "--subdivide-error 0 to re-enable it (default 0.1).",
            file=sys.stderr, flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vff", description="VFF Slicer — interactive visualizer")
    parser.add_argument("stl", nargs="?", help="Path to STL file (defaults to ./propeller_fixed_flat.stl).")
    parser.add_argument(
        "--volume", default="250x250x250",
        help="Build volume as X x Y x Z in mm (default: 250x250x250).",
    )
    parser.add_argument("--pitch", type=float, default=1.0, help="Initial voxel pitch in mm (default: 1.0).")
    parser.add_argument(
        "--max-tilt", type=float, default=30.0,
        help="Max nozzle tilt from vertical in degrees (default 30). Set at startup, not changed dynamically.",
    )
    parser.add_argument(
        "--smooth-sigma", type=float, default=2.0,
        help="Gaussian sigma (in voxels), default 2.0. 3d smooths the DISPLACEMENT "
             "field (smoother mesh, fewer folds; softens domes — pair with "
             "--subdivide-error to keep them, or lower sigma for a sharper dome). "
             "z-only smooths the DEPTH field. Lower = sharper; higher = smoother/weaker.",
    )
    parser.add_argument(
        "--depth-method", choices=["vectors", "harmonic", "fmm", "dijkstra"], default="vectors",
        help="Inside-model depth computation. 'vectors' (default): integrate the CLAMPED "
             "growth vector field into a potential whose level sets are the layer surfaces "
             "(surfaces perpendicular to the growth direction; tilt clamp shapes them). "
             "'harmonic': solve Laplace Δφ=0 (φ=0 on the bed, φ=height on top voxels) — "
             "level sets are exact normals to ∇φ, never close inside the part (no shells "
             "at a mushroom cap), and don't inherit the geometry's concavity; NO tilt clamp "
             "(overhangs may exceed max-tilt). 'fmm' (Eikonal, C1 smooth, requires scikit-fmm) "
             "or 'dijkstra' (discrete shortest path, C0) — both ignore the clamp and follow "
             "raw geodesic depth. In --deform-mode 3d this selects the growth DIRECTION the "
             "map rotates to vertical: 'vectors' = clamped BFS vectors (default), 'harmonic' = "
             "∇φ (curl-free, no clamp). Must MATCH between the sliced export and the inverse.",
    )
    parser.add_argument(
        "--deform-mode", choices=["z-only", "3d"], default="z-only",
        help="Deformation model for --export and --gcode-in. 'z-only' (default, "
             "legacy): only Z moves, XY frozen — shears in-plane distances on "
             "tilted layers. '3d': full straightening — a Poisson/ARAP map that "
             "rotates the growth direction to vertical, moving ALL axes (preserves "
             "in-plane distances, much better volume; see vff/deform3d.py). 3d "
             "uses --smooth-sigma (Gaussian smoothing of the displacement field "
             "— removes voxel-scale surface waviness and makes the map injective; "
             "MUST match between --export and the inverse) but ignores "
             "--dz-per-layer/--dz-auto-fit/--depth-method. Note: the interactive "
             "viewer is z-only regardless.",
    )
    parser.add_argument(
        "--subdivide-error", type=float, default=0.1, metavar="MM",
        help="(--deform-mode 3d --export only) Refine the mesh before deforming "
             "until the worst per-face deformation error drops below this many "
             "mm (default 0.1; 0 = off). Fixes flat regions defined by few large "
             "triangles (e.g. a bore ceiling) that otherwise stay flat instead "
             "of following the curved deformation. Capped at ~2M faces. See "
             "--remesh for the refinement method.",
    )
    parser.add_argument(
        "--remesh", choices=["adaptive", "uniform"], default="adaptive",
        help="(--subdivide-error) Refinement method. 'adaptive' (default): "
             "Rivara longest-edge bisection — conforming/crack-free, refines "
             "only the curved faces (~15× fewer faces than uniform for the same "
             "quality, e.g. propeller 562k → 36k). 'uniform': trimesh 1→4 "
             "subdivide of every face each pass — simpler but far heavier.",
    )
    parser.add_argument(
        "--extrusion-comp", action=argparse.BooleanOptionalAction, default=True,
        help="(--deform-mode 3d only) Rescale each G-code segment's E for the "
             "deformation so stretched/compressed layers get the right amount of "
             "material. Requires relative E (M83); absolute E left uncompensated. "
             "Default on; --no-extrusion-comp to disable.",
    )
    parser.add_argument(
        "--z-slowdown", type=float, default=1.0, metavar="FACTOR",
        help="(gcode transform) Gently slow the feedrate on steep non-planar "
             "moves (1.0 = off). F is scaled from ×1 on flat moves down to "
             "×FACTOR at a ~30°-tilted move and steeper, so the firmware's Z "
             "planner isn't fighting a feedrate aimed straight up. e.g. 0.5 = "
             "halve F on the steepest parts. A soft ease for the transition; for "
             "an actual bound on Z velocity use --max-z-speed.",
    )
    parser.add_argument(
        "--max-z-speed", type=float, default=15.0, metavar="MM/S",
        help="(gcode transform) HARD cap on the Z-axis velocity component "
             "(default 15 mm/s; 0 = off). On a curved move the Z speed is "
             "F·|dz|/L; F is recomputed per segment so it never exceeds this, "
             "i.e. the firmware's Z clamp (Klipper max_z_velocity) is applied in "
             "the toolpath itself so the planner sees honest feedrates instead "
             "of silently dragging the whole move down to obey Z. Flat moves are "
             "untouched. Composes with --z-slowdown (the lower F wins).",
    )
    parser.add_argument(
        "--extrusion-comp-mode", choices=["vertical", "volume"], default="vertical",
        help="Extrusion-comp model. 'vertical' (default, 3-axis vertical nozzle): "
             "scale by the layer-height squish ∂orig_z/∂def_z — fixed road width, "
             "so only the layer spacing matters; reduces E where layers compress. "
             "'volume' (4/5-axis, S4-style): scale by 1/det(JΦ) — material-"
             "conservative, for a tilting nozzle whose road deforms in all axes. "
             "On the propeller they differ ~12% and opposite sign in the bulk.",
    )
    parser.add_argument(
        "--cool-overhangs", action=argparse.BooleanOptionalAction, default=True,
        help="(--deform-mode 3d, --gcode-direction inverse) Re-detect overhangs/"
             "bridges on the ORIGINAL-space toolpath and force the fan to "
             "--cool-fan there. The slicer schedules cooling from the deformed, "
             "flat mesh; after the inverse, surfaces it saw as flat can hang over "
             "a void in the real part. Default on; --no-cool-overhangs to disable "
             "(e.g. ABS/ASA, or to skip the extra mesh-containment pass).",
    )
    parser.add_argument(
        "--cool-fan-min", type=int, default=128, metavar="0-255",
        help="(--cool-overhangs) Fan PWM at the LIGHTEST detected overhang "
             "(default 128). The fan ramps linearly from here to --cool-fan-max "
             "with overhang severity (like a slicer's per-overlap fan curve). "
             "Only ever raises the fan above the slicer's own value.",
    )
    parser.add_argument(
        "--cool-fan-max", type=int, default=255, metavar="0-255",
        help="(--cool-overhangs) Fan PWM at a full bridge / worst overhang "
             "(default 255 = full).",
    )
    parser.add_argument(
        "--cool-speed", type=float, default=20.0, metavar="MM/S",
        help="(--cool-overhangs) Print speed at a full bridge / worst overhang "
             "(default 20 mm/s; 0 = off). The feedrate twin of the cooling fan: "
             "on the detected overhang moves F is ramped down from the slicer's "
             "speed (degree 0) to this at a full bridge (degree 1), giving the "
             "freshly-laid road time to set over the void. Only ever lowers F, "
             "and composes with --z-slowdown / --max-z-speed (the lowest F wins).",
    )
    parser.add_argument(
        "--cool-probe", type=float, default=0.8, metavar="MM",
        help="(--cool-overhangs) Depth (mm) of the downward support-probe "
             "column. Severity = fraction of that column (4 samples) that is "
             "air below the point: 0 = solid right below (no boost), 1 = air all "
             "the way (bridge → --cool-fan-max). Default 0.8 (~a few layers).",
    )
    parser.add_argument(
        "--cool-min-z", type=float, default=0.6, metavar="MM",
        help="(--cool-overhangs) Never flag points within this height of the "
             "plate (the bed supports them). Default 0.6 mm.",
    )
    parser.add_argument(
        "--export", metavar="PATH",
        help="Save the deformed mesh to PATH right after startup (.stl, .ply, .obj, .glb — any "
             "format trimesh supports). Computes voxels + growth + deform first.",
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Skip the interactive viewer. Useful with --export for batch use.",
    )
    parser.add_argument(
        "--section-xz", metavar="PATH",
        help="Render an XZ-plane cross-section (normal +Y) of the growth layer "
             "surfaces and save it to PATH (.png). Only the surfaces' cut curves "
             "are drawn, orthographic, looking down Y. Headless/off-screen. "
             "Combine with --no-viewer for a pure batch figure.",
    )
    parser.add_argument(
        "--section-y", type=float, default=None,
        help="Y coordinate (mm) of the --section-xz cutting plane. "
             "Default: the model's Y centre.",
    )
    parser.add_argument(
        "--gcode-in", metavar="PATH",
        help="Read a G-code file and transform every XYZ point through the depth-field "
             "deformation (see --gcode-direction). Skips the viewer; for batch slicer feed.",
    )
    parser.add_argument(
        "--gcode-out", metavar="PATH",
        help="Output path for --gcode-in. Defaults to <input>.transformed.gcode.",
    )
    parser.add_argument(
        "--gcode-direction", choices=["forward", "inverse"], default="forward",
        help="forward (default): apply deform_mesh's map to G-code — planar slicer output "
             "on a flat/slicer-friendly mesh -> non-planar G-code following the depth-field "
             "layers. inverse: apply deform_mesh's inverse — planar slicer output on a "
             "pre-deformed mesh -> G-code in the original mesh's coord system.",
    )
    # Back-compat aliases for the previous flag names.
    parser.add_argument("--backtransform-in", dest="gcode_in", help=argparse.SUPPRESS)
    parser.add_argument("--backtransform-out", dest="gcode_out", help=argparse.SUPPRESS)
    parser.add_argument(
        "--subdiv-mm", type=float, default=0.5,
        help="Backtransform: split G1 moves longer than this (in deformed-space mm) into "
             "pieces so the curved original-space path is followed. Default 0.5.",
    )
    parser.add_argument(
        "--dz-per-layer", type=float, default=None,
        help="Map factor: deformed_z = depth(x,y,z) * dz_per_layer + bed (+ bed-blend). "
             "If unset and --dz-auto-fit is off, falls back to pitch (1 voxel = 1 mm by "
             "default). Use this to control how much the deformation actually stretches Z.",
    )
    parser.add_argument(
        "--dz-auto-fit", action="store_true",
        help="Pick dz_per_layer automatically so the FORWARD-deformed Z extent matches the "
             "input STL's Z extent. I.e. blade tips end up at the same height the slicer "
             "thought they'd be at, but with non-planar layer paths in between. "
             "Overrides --dz-per-layer if both given.",
    )
    parser.add_argument(
        "--clip-to", metavar="STL_PATH",
        help="G-code transform mode: drop extrusion outside the given STL's "
             "interior. Each subdivided G1 piece is tested with mesh.contains(); "
             "inside-pieces keep their E, outside-pieces become G0 (no E). "
             "Use to print a curved-bottom target from gcode that was sliced "
             "for a flat-bottomed variant — material in the filled-base region "
             "is removed. Visual verification only — leaves unsupported "
             "geometry; real prints need supports/transitions.",
    )
    parser.add_argument(
        "--conform-to", metavar="STL_PATH",
        help="G-code transform mode: skip the depth-field deformation and "
             "instead lift each gcode point's Z by the bottom-surface height "
             "of the given target STL at that (X, Y), via ray-cast from below. "
             "Use case: 'print this curved-bottom mesh from gcode that was "
             "sliced for a flat-bottomed variant (target + added flat slab)'. "
             "Preserves slab thickness, blade thickness and hub height — only "
             "Z-shifts each column rigidly. Ignores --dz-per-layer, "
             "--dz-auto-fit, --max-tilt, --smooth-sigma, --depth-method.",
    )
    parser.add_argument(
        "--no-gcode-align", action="store_true",
        help="Disable XY auto-alignment of the depth field to the gcode's XY bounds. "
             "By default the gcode-transform path quick-scans the G-code XY range and "
             "translates the STL so its XY bbox centre matches the gcode's — necessary "
             "because slicers position the model on their own bed (e.g. PrusaSlicer 220x220) "
             "which won't match our --volume centre by default. Disable only if you know "
             "the STL is already in slicer-coords.",
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Backtransform: parallel worker count for the invert step. "
             "-1 = auto (all cores, but only when point count >5 M; below that the "
             "vectorised NumPy single-pass beats the multiprocessing spawn overhead). "
             "0 / 1 = force single-process. N>1 = force N workers.",
    )
    parser.add_argument(
        "--preview-gcode", metavar="PATH",
        help="Load a G-code file into the viewer overlay (extrusion polylines, coloured by Z). "
             "Toggle in viewer with P. PrusaSlicer's own preview won't show non-planar layers; "
             "this one will.",
    )
    args = parser.parse_args(argv)

    # Fast path: --gcode-in + (--conform-to | --clip-to) needs neither a build
    # volume nor the positional STL (depth field is bypassed). Skip the
    # expensive voxelize/Viewer setup entirely.
    if args.gcode_in and (args.conform_to or args.clip_to):
        from .backtransform import (
            clip_gcode_file, quick_gcode_xy_bounds, surface_offset_gcode_file,
        )
        mode = "clip" if args.clip_to else "conform"
        target_arg = args.clip_to if args.clip_to else args.conform_to
        target_path = Path(target_arg)
        if not target_path.exists():
            print(f"--{mode}-to STL not found: {target_path}", file=sys.stderr)
            return 1
        out_path = args.gcode_out
        if not out_path:
            default_suffix = ".clipped.gcode" if mode == "clip" else ".conformed.gcode"
            out_path = str(Path(args.gcode_in).with_suffix(default_suffix))
        xy_center = None
        align_info = ""
        if not args.no_gcode_align:
            xy_min, xy_max = quick_gcode_xy_bounds(args.gcode_in)
            if np.isfinite(xy_min).all() and np.isfinite(xy_max).all():
                xy_center = (
                    0.5 * (xy_min[0] + xy_max[0]),
                    0.5 * (xy_min[1] + xy_max[1]),
                )
                align_info = (
                    f"  align→gcode  : XY range [{xy_min[0]:.1f}, {xy_max[0]:.1f}] x "
                    f"[{xy_min[1]:.1f}, {xy_max[1]:.1f}], centre "
                    f"({xy_center[0]:.1f}, {xy_center[1]:.1f})\n"
                )
        banner_title = (
            "G-code clip to mesh interior" if mode == "clip"
            else "G-code surface-offset conform"
        )
        print(
            f"{banner_title}: {args.gcode_in} -> {out_path}\n"
            f"  target STL    : {target_path}\n"
            f"  subdiv-mm     : {args.subdiv_mm} mm\n"
            f"{align_info}",
            end="",
            flush=True,
        )
        print(flush=True)
        if mode == "clip":
            clip_gcode_file(
                args.gcode_in, out_path, str(target_path),
                xy_center=xy_center, subdiv_mm=args.subdiv_mm,
            )
        else:
            surface_offset_gcode_file(
                args.gcode_in, out_path, str(target_path),
                xy_center=xy_center, subdiv_mm=args.subdiv_mm,
            )
        return 0

    # Fast path: full-3D export and/or gcode transform. The 3D deformation
    # (vff/deform3d.py) is fully headless — no Viewer / pyvista — so handle it
    # before importing the GUI stack. (The interactive viewer is z-only, so
    # there's nothing 3D to show there anyway.)
    if args.deform_mode == "3d" and (args.export or args.gcode_in):
        stl_path = Path(args.stl) if args.stl else Path.cwd() / "propeller_fixed_flat.stl"
        if not stl_path.exists():
            print(f"STL not found: {stl_path}", file=sys.stderr)
            return 1
        try:
            x, y, z = (float(v) for v in args.volume.lower().split("x"))
        except ValueError:
            print(f"Bad --volume '{args.volume}'. Expected e.g. 250x250x250.", file=sys.stderr)
            return 2
        if args.export:
            _run_3d_export(args, stl_path, x, y, z)
        if args.gcode_in:
            return _run_gcode_transform(args, stl_path, x, y, z)
        return 0

    # Fast path: depth-field G-code transform (forward/inverse) needs the depth
    # field but NOT the interactive Viewer / pyvista. Handle it before importing
    # the GUI stack so batch/headless slicing works without a display. Combined
    # with viewer-only flags (--export / --section-xz / --preview-gcode) it falls
    # through to the full path below.
    if args.gcode_in and not (args.export or args.section_xz or args.preview_gcode):
        stl_path = Path(args.stl) if args.stl else Path.cwd() / "propeller_fixed_flat.stl"
        if not stl_path.exists():
            print(f"STL not found: {stl_path}", file=sys.stderr)
            return 1
        try:
            x, y, z = (float(v) for v in args.volume.lower().split("x"))
        except ValueError:
            print(f"Bad --volume '{args.volume}'. Expected e.g. 250x250x250.", file=sys.stderr)
            return 2
        return _run_gcode_transform(args, stl_path, x, y, z)

    from .build_volume import BuildVolume
    from .mesh_io import load_and_place
    from .viewer import Viewer

    stl_path = Path(args.stl) if args.stl else Path.cwd() / "propeller_fixed_flat.stl"
    if not stl_path.exists():
        print(f"STL not found: {stl_path}", file=sys.stderr)
        return 1

    try:
        x, y, z = (float(v) for v in args.volume.lower().split("x"))
    except ValueError:
        print(f"Bad --volume '{args.volume}'. Expected e.g. 250x250x250.", file=sys.stderr)
        return 2

    volume = BuildVolume.of(x, y, z)
    print(f"Loading {stl_path} into build volume {x:.0f}x{y:.0f}x{z:.0f} mm ...", flush=True)
    mesh = load_and_place(str(stl_path), volume)
    print(
        f"Mesh placed: triangles={len(mesh.faces):,}, "
        f"bounds={mesh.bounds.tolist()}",
        flush=True,
    )

    viewer = Viewer(
        mesh, volume,
        initial_pitch=args.pitch,
        max_tilt_deg=args.max_tilt,
        smooth_sigma=args.smooth_sigma,
        depth_method=args.depth_method,
        # Honour --dz-per-layer for the exported mesh (was silently ignored).
        dz_per_layer=args.dz_per_layer,
    )

    if args.export:
        viewer.save_deformed(args.export)
        if args.no_viewer and not args.section_xz:
            return 0

    if args.section_xz:
        from .section import save_xz_section
        print(f"Rendering XZ growth-surface section -> {args.section_xz} ...", flush=True)
        n_pts = save_xz_section(
            mesh, args.section_xz,
            pitch=args.pitch,
            max_tilt_deg=args.max_tilt,
            smooth_sigma=args.smooth_sigma,
            depth_method=args.depth_method,
            section_y=args.section_y,
        )
        if n_pts == 0:
            print(
                "  WARNING: section plane hit no surfaces — check --section-y.",
                flush=True,
            )
        else:
            print(f"  saved ({n_pts:,} section points).", flush=True)
        if args.no_viewer:
            return 0

    if args.preview_gcode:
        viewer.load_gcode_preview(args.preview_gcode)

    if args.gcode_in:
        return _run_gcode_transform(args, stl_path, x, y, z)

    print(
        "Viewer ready. Hotkeys: M/V/B  G/C/N/H/D/O  [ / ]  Up/Down  F5",
        flush=True,
    )
    print(
        "  M mesh  V voxel-shell  B re-voxel | G growth  C voxels  N vectors  H surface  D deformed  O export",
        flush=True,
    )
    viewer.show()
    print("Viewer closed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Real failures — but not SystemExit/KeyboardInterrupt, which fall
        # through to the interpreter's normal exit path.
        traceback.print_exc()
        sys.stderr.flush()
        raise
