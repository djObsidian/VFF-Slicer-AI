import argparse
import faulthandler
import sys
import traceback
from pathlib import Path

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
        from .backtransform import BackTransform, backtransform_gcode_file
        out_path = args.gcode_out
        if not out_path:
            from pathlib import Path as _P
            in_p = _P(args.gcode_in)
            suffix = ".nonplanar.gcode" if args.gcode_direction == "forward" else ".planar.gcode"
            out_path = str(in_p.with_suffix(suffix))
        print(
            f"G-code transform ({args.gcode_direction}): {args.gcode_in} -> {out_path}\n"
            f"  STL          : {stl_path}\n"
            f"  volume       : {x:.0f}x{y:.0f}x{z:.0f} mm\n"
            f"  pitch        : {args.pitch} mm\n"
            f"  max-tilt     : {args.max_tilt} deg\n"
            f"  smooth-sigma : {args.smooth_sigma}\n"
            f"  depth-method : {args.depth_method}\n"
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
