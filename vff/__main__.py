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
        "--backtransform-in", metavar="PATH",
        help="Read a G-code file produced by a planar slicer on the deformed mesh and "
             "inverse-transform every XYZ point back to the original (non-planar) space. "
             "Writes the result to --backtransform-out (default: input with '.nonplanar.gcode' suffix). "
             "Skips the viewer.",
    )
    parser.add_argument(
        "--backtransform-out", metavar="PATH",
        help="Output path for --backtransform-in. Defaults to <input>.nonplanar.gcode.",
    )
    parser.add_argument(
        "--subdiv-mm", type=float, default=0.5,
        help="Backtransform: split G1 moves longer than this (in deformed-space mm) into "
             "pieces so the curved original-space path is followed. Default 0.5.",
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

    if args.backtransform_in:
        from .backtransform import BackTransform, backtransform_gcode_file
        out_path = args.backtransform_out
        if not out_path:
            from pathlib import Path as _P
            in_p = _P(args.backtransform_in)
            out_path = str(in_p.with_suffix(".nonplanar.gcode"))
        print(
            f"Backtransform: {args.backtransform_in} -> {out_path}\n"
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
        backtransform_gcode_file(args.backtransform_in, out_path, bt, subdiv_mm=args.subdiv_mm)
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
