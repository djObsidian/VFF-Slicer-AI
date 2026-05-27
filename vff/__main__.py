import argparse
import faulthandler
import sys
import traceback
from pathlib import Path

import numpy as np

# --- Surface every kind of failure to the console BEFORE we import VTK ---
# 1. Unbuffered stdio so prints survive a hard crash.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vff", description="VFF Slicer — interactive visualizer")
    parser.add_argument("stl", nargs="?", help="Path to STL file (defaults to ./propeller.stl).")
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
        help="Gaussian sigma (in voxels) applied to the depth field before deformation/surfaces "
             "(default 2.0). Lower = stronger / sharper deformation; higher = smoother but weaker.",
    )
    parser.add_argument(
        "--depth-method", choices=["fmm", "dijkstra"], default="fmm",
        help="Inside-model depth computation: 'fmm' (Eikonal, C1 smooth, requires scikit-fmm) "
             "or 'dijkstra' (discrete shortest path, C0). Default 'fmm'.",
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

    from .build_volume import BuildVolume
    from .mesh_io import load_and_place
    from .viewer import Viewer

    stl_path = Path(args.stl) if args.stl else Path.cwd() / "propeller.stl"
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
    )

    if args.export:
        viewer.save_deformed(args.export)
        if args.no_viewer:
            return 0

    if args.preview_gcode:
        viewer.load_gcode_preview(args.preview_gcode)

    if args.gcode_in:
        from .backtransform import BackTransform, backtransform_gcode_file, quick_gcode_xy_bounds
        out_path = args.gcode_out
        if not out_path:
            from pathlib import Path as _P
            in_p = _P(args.gcode_in)
            suffix = ".nonplanar.gcode" if args.gcode_direction == "forward" else ".planar.gcode"
            out_path = str(in_p.with_suffix(suffix))

        # First: quick-scan the gcode for its XY range so we can align the
        # depth field to the slicer's model placement. PrusaSlicer / Cura
        # don't use our --volume centre.
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

        # Auto-fit picks dz so the forward-deformed Z extent matches the STL's
        # Z extent. Computed by building the depth field once, taking its max
        # over model voxels, then dz = z_extent / depth_max.
        chosen_dz = args.dz_per_layer
        autofit_info = ""
        if args.dz_auto_fit:
            import numpy as _np
            import trimesh as _trimesh
            from .deform import smoothed_depth_field
            from .growth import compute_growth
            from .voxelize import voxelize_solid

            _m = _trimesh.load(str(stl_path), force="mesh")
            if xy_center is not None:
                _mxy = 0.5 * (_m.bounds[0, :2] + _m.bounds[1, :2])
                _m.apply_translation([
                    xy_center[0] - float(_mxy[0]),
                    xy_center[1] - float(_mxy[1]),
                    -float(_m.bounds[0, 2]),
                ])
            else:
                from .build_volume import BuildVolume as _BV
                _vol = _BV.of(x, y, z)
                _m = load_and_place(str(stl_path), _vol)
            _vg = voxelize_solid(_m, pitch=args.pitch)
            _gr = compute_growth(_vg, max_tilt_deg=args.max_tilt)
            _f = smoothed_depth_field(
                _gr, sigma=args.smooth_sigma, method=args.depth_method, outside_mode="extend"
            )
            _depth_max = float(_np.nanmax(_f[_gr.step >= 0])) if (_gr.step >= 0).any() else 1.0
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
            direction=args.gcode_direction,
        )
        return 0

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
