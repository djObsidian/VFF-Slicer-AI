"""Quick overlay render: propeller.stl + a gcode polyline."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from vff.gcode_preview import parse_gcode


def main(gcode_path: str, target_path: str = "propeller.stl",
         xy_center=(110.0, 110.0), out_prefix: str = "_cmp") -> None:
    m = trimesh.load(target_path, force="mesh")
    m.apply_translation([0.0, 0.0, -float(m.bounds[0, 2])])
    mxy = 0.5 * (m.bounds[0, :2] + m.bounds[1, :2])
    m.apply_translation([
        float(xy_center[0]) - float(mxy[0]),
        float(xy_center[1]) - float(mxy[1]),
        0.0,
    ])
    print(f"target {target_path}: bounds={m.bounds.tolist()}")

    pv_mesh = pv.PolyData(
        np.asarray(m.vertices, dtype=np.float64),
        faces=np.column_stack([
            np.full(len(m.faces), 3, dtype=np.int64),
            np.asarray(m.faces, dtype=np.int64),
        ]).ravel(),
    )

    data = parse_gcode(gcode_path)
    ep = data["extrusion_points"]
    el = data["extrusion_lines"]
    print(f"gcode {gcode_path}: {data['n_extrusion_moves']:,} extrusion moves, "
          f"Z=[{ep[:,2].min():.2f},{ep[:,2].max():.2f}]")
    gpd = pv.PolyData(ep, lines=el)
    gpd.cell_data["z"] = data["extrusion_step"]

    for view_name, cam_pos in [
        ("iso", "iso"),
        ("side", [(110, -100, 12), (110, 110, 12), (0, 0, 1)]),
        ("top", [(110, 110, 100), (110, 110, 12), (0, 1, 0)]),
    ]:
        p = pv.Plotter(off_screen=True, window_size=(960, 720))
        p.set_background("#1e1e22", top="#3a3a44")
        p.add_mesh(pv_mesh, color="#6a90c0", opacity=0.35, lighting=True, name="target")
        p.add_mesh(gpd, scalars="z", cmap="plasma", line_width=1.5,
                   show_scalar_bar=True, scalar_bar_args={"title": "gcode Z"})
        if cam_pos == "iso":
            p.camera_position = "iso"
            p.camera.zoom(1.2)
        else:
            p.camera_position = cam_pos
        out = f"{out_prefix}_{view_name}.png"
        p.screenshot(out)
        print(f"  -> {out}")
        p.close()


if __name__ == "__main__":
    gcode = sys.argv[1] if len(sys.argv) > 1 else "propeller_inverse.gcode"
    target = sys.argv[2] if len(sys.argv) > 2 else "propeller.stl"
    prefix = sys.argv[3] if len(sys.argv) > 3 else "_cmp"
    main(gcode, target, out_prefix=prefix)
