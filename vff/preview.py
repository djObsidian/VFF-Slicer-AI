"""Standalone G-code viewer. Bed + build-volume wireframe + extrusion
polylines coloured by Z. Useful for non-planar G-code (any 3-axis output
that PrusaSlicer's own preview can't render: curved layers, very fine
Z-steps, post-processed paths, etc.).

Usage:

    python -m vff.preview deformed.gcode
    python -m vff.preview my_nonplanar.gcode --volume 250x250x250
    python -m vff.preview my.gcode --show-travel

Hotkeys in the window:
    T : toggle travel moves (off by default — they swamp the picture)
    E : toggle extrusion moves
    R : VTK reset camera
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from .build_volume import BuildVolume
from .gcode_preview import parse_gcode


def _make_bed_grid_lines(volume_size: np.ndarray, step: float = 10.0) -> pv.PolyData:
    sx, sy, _ = volume_size
    xs = np.arange(0.0, sx + step * 0.5, step)
    ys = np.arange(0.0, sy + step * 0.5, step)
    points: list[list[float]] = []
    cells: list[int] = []
    idx = 0
    for x in xs:
        points.append([x, 0.0, 0.0])
        points.append([x, sy, 0.0])
        cells.extend([2, idx, idx + 1])
        idx += 2
    for y in ys:
        points.append([0.0, y, 0.0])
        points.append([sx, y, 0.0])
        cells.extend([2, idx, idx + 1])
        idx += 2
    return pv.PolyData(np.asarray(points, dtype=np.float64), lines=np.asarray(cells, dtype=np.int64))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vff.preview",
        description="Render a G-code file in 3D (extrusion polylines coloured by Z).",
    )
    parser.add_argument("gcode", help="Path to the G-code file.")
    parser.add_argument(
        "--volume", default="250x250x250",
        help="Build volume X x Y x Z mm (default 250x250x250).",
    )
    parser.add_argument(
        "--show-travel", action="store_true",
        help="Show travel (non-extrusion) moves too. Off by default — they "
             "tend to swamp the picture. Toggleable with T at runtime.",
    )
    parser.add_argument(
        "--cmap", default="plasma",
        help="Matplotlib colormap for extrusion-Z (default plasma). "
             "Try viridis, inferno, turbo.",
    )
    parser.add_argument(
        "--line-width", type=float, default=2.0,
        help="Extrusion line width in pixels (default 2).",
    )
    args = parser.parse_args(argv)

    gcode_path = Path(args.gcode)
    if not gcode_path.exists():
        print(f"G-code not found: {gcode_path}", file=sys.stderr)
        return 1

    try:
        vx, vy, vz = (float(v) for v in args.volume.lower().split("x"))
    except ValueError:
        print(f"Bad --volume '{args.volume}'. Expected e.g. 250x250x250.", file=sys.stderr)
        return 2
    volume = BuildVolume.of(vx, vy, vz)

    print(f"Parsing {gcode_path} …", flush=True)
    import time as _time
    t0 = _time.perf_counter()
    data = parse_gcode(gcode_path)
    dt_ms = (_time.perf_counter() - t0) * 1000.0
    n_ext = data["n_extrusion_moves"]
    n_trv = data["n_travel_moves"]
    ext_pts = data["extrusion_points"]
    if ext_pts.shape[0] > 0:
        zmin = float(ext_pts[:, 2].min())
        zmax = float(ext_pts[:, 2].max())
        xmin = float(ext_pts[:, 0].min())
        xmax = float(ext_pts[:, 0].max())
        ymin = float(ext_pts[:, 1].min())
        ymax = float(ext_pts[:, 1].max())
    else:
        zmin = zmax = 0.0
        xmin = xmax = ymin = ymax = 0.0
    print(
        f"  {n_ext:,} extrusion + {n_trv:,} travel moves  ({dt_ms:.0f} ms parse)\n"
        f"  bounds: X[{xmin:.1f},{xmax:.1f}] Y[{ymin:.1f},{ymax:.1f}] Z[{zmin:.2f},{zmax:.2f}] mm",
        flush=True,
    )

    plotter = pv.Plotter(title=f"vff.preview — {gcode_path.name}", window_size=(1280, 800))
    plotter.set_background("#1e1e22", top="#3a3a44")
    try:
        pv.set_new_attribute(plotter, "pickpoint", None)
    except Exception:
        pass

    sx, sy, sz = volume.size
    bed = pv.Plane(
        center=(sx * 0.5, sy * 0.5, 0.0), direction=(0, 0, 1),
        i_size=sx, j_size=sy, i_resolution=1, j_resolution=1,
    )
    plotter.add_mesh(bed, color="#dddddd", opacity=0.18, lighting=False, name="bed", pickable=False)
    plotter.add_mesh(
        _make_bed_grid_lines(volume.size, step=10.0),
        color="#888888", line_width=1, lighting=False, name="bed_grid", pickable=False,
    )
    plotter.add_mesh(
        pv.Box(bounds=volume.bounds).extract_feature_edges(),
        color="#4169e1", line_width=2, lighting=False, name="volume_box", pickable=False,
    )
    plotter.add_axes(interactive=False)

    ext_actor = None
    if ext_pts.shape[0] > 0:
        pd = pv.PolyData(ext_pts, lines=data["extrusion_lines"])
        pd.cell_data["z_mid"] = data["extrusion_step"]
        ext_actor = plotter.add_mesh(
            pd, scalars="z_mid", cmap=args.cmap, line_width=float(args.line_width),
            show_scalar_bar=True, scalar_bar_args={"title": "Z mid (mm)"},
            lighting=False, name="extrusion",
        )

    trv_actor = None
    trv_pts = data["travel_points"]
    if trv_pts.shape[0] > 0:
        pd_t = pv.PolyData(trv_pts, lines=data["travel_lines"])
        trv_actor = plotter.add_mesh(
            pd_t, color="#666666", line_width=1, lighting=False, opacity=0.4, name="travel",
        )
        if trv_actor is not None:
            trv_actor.SetVisibility(bool(args.show_travel))

    show_ext = {"on": True}
    show_trv = {"on": bool(args.show_travel)}

    def toggle_extrusion():
        show_ext["on"] = not show_ext["on"]
        if ext_actor is not None:
            ext_actor.SetVisibility(show_ext["on"])
        plotter.render()

    def toggle_travel():
        show_trv["on"] = not show_trv["on"]
        if trv_actor is not None:
            trv_actor.SetVisibility(show_trv["on"])
        plotter.render()

    plotter.add_key_event("e", toggle_extrusion)
    plotter.add_key_event("t", toggle_travel)

    # Hud: title overlay so user remembers the keys.
    plotter.add_text(
        f"vff.preview — {gcode_path.name}\n"
        f"{n_ext:,} extrusion  {n_trv:,} travel\n"
        f"Z [{zmin:.2f}, {zmax:.2f}] mm\n"
        "[E] extrusion  [T] travel  [R] reset cam",
        position="upper_left", font_size=10, color="#eeeeee", font="courier", shadow=True,
    )

    # Frame the camera on whatever geometry the file actually has, not the
    # whole build volume (so previews of tiny test parts aren't lost in space).
    if ext_pts.shape[0] > 0:
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        diag = max(xmax - xmin, ymax - ymin, zmax - zmin, 10.0) * 2.0
        plotter.camera_position = [
            (cx + diag, cy - diag, cz + diag),
            (cx, cy, cz),
            (0, 0, 1),
        ]
    else:
        plotter.camera_position = "iso"
    plotter.reset_camera_clipping_range()

    plotter.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
