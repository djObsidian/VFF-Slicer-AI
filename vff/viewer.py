"""Interactive PyVista visualizer for VFF Slicer.

Scene composition (all in world coords, Z up):
- Build plate at Z=0
- Bed grid (10 mm lines by default)
- Build volume wireframe box
- World-origin axis triad
- Loaded mesh (centered on bed)
- Voxel grid (built on demand)

Hotkeys:
- M : toggle mesh visibility
- V : toggle voxels (builds them the first time if needed)
- B : (re)build voxels at current pitch
- [ / ] : decrease / increase pitch by 25%
- R : reset camera
- W : wireframe (built into VTK)
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Callable

import numpy as np
import pyvista as pv
import trimesh

from .build_volume import BuildVolume
from .voxelize import VoxelGrid, voxelize_solid


def _log(msg: str) -> None:
    """Force-flushed stderr write so messages survive a hard VTK crash."""
    print(msg, file=sys.stderr, flush=True)


def _safe(label: str, fn: Callable[[], None]) -> Callable[[], None]:
    """Wrap a no-arg callback so any exception is logged with full traceback
    and the entry/exit is logged so a native crash can be pinned to a specific
    callback."""
    def wrapped() -> None:
        _log(f"[vff] -> {label}")
        try:
            fn()
        except BaseException:
            _log(f"[vff] ERROR in callback '{label}':")
            _log(traceback.format_exc())
        else:
            _log(f"[vff] <- {label} ok")
    return wrapped


_BED_GRID_STEP_MM = 10.0
_BED_COLOR = "#dddddd"
_BED_GRID_COLOR = "#888888"
_VOLUME_EDGE_COLOR = "#4169e1"
_MESH_COLOR = "#88aacc"
_VOXEL_COLOR = "#ff8855"
_BACKGROUND = "#1e1e22"
_BACKGROUND_TOP = "#3a3a44"
_HUD_COLOR = "#eeeeee"


def trimesh_to_pv(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.column_stack(
        [np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(mesh.vertices, faces=faces)


def voxelgrid_to_image_data(vg: VoxelGrid) -> pv.ImageData:
    """Wrap a VoxelGrid as a pv.ImageData with `filled` cell scalar."""
    nx, ny, nz = vg.matrix.shape
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),  # point dims = cell dims + 1
        spacing=(vg.pitch, vg.pitch, vg.pitch),
        origin=tuple(vg.origin),
    )
    # VTK cell ordering: X fastest, then Y, then Z. numpy 'F' order matches.
    grid.cell_data["filled"] = vg.matrix.astype(np.uint8).flatten(order="F")
    return grid


def _make_bed_grid_lines(size: np.ndarray, step: float = _BED_GRID_STEP_MM) -> pv.PolyData:
    sx, sy, _ = size
    xs = np.arange(0.0, sx + step * 0.5, step)
    ys = np.arange(0.0, sy + step * 0.5, step)

    points = []
    cells = []
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


class Viewer:
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        volume: BuildVolume,
        initial_pitch: float = 1.0,
    ) -> None:
        self.mesh = mesh
        self.volume = volume
        self.pitch = float(initial_pitch)

        self.plotter = pv.Plotter(title="VFF Slicer — Visualizer", window_size=(1280, 800))
        self.plotter.set_background(_BACKGROUND, top=_BACKGROUND_TOP)
        # Workaround PyVista 0.48 bug: `left_button_down` callback tries to
        # set `self.pickpoint` on the Plotter, but Plotter rejects new public
        # attributes — every mouse click would emit a UserWarning. Pre-register
        # the attribute via PyVista's own escape hatch.
        try:
            pv.set_new_attribute(self.plotter, "pickpoint", None)
        except Exception:
            pass

        self.pv_mesh = trimesh_to_pv(mesh)

        self.mesh_actor: pv.Actor | None = None
        self.voxel_actor: pv.Actor | None = None
        self.voxel_grid: VoxelGrid | None = None
        self._last_voxelize_ms: float | None = None
        self._hud_actor = None  # persistent text actor; updated via SetInput

        self.show_mesh = True
        self.show_voxels = False

        self._build_static_scene()
        self._add_mesh_actor()
        self._init_hud()
        self._bind_keys()
        self._refresh_hud()

        self.plotter.add_axes(interactive=False)
        self._set_default_view()

    # ----- scene -----

    def _build_static_scene(self) -> None:
        sx, sy, sz = self.volume.size

        bed = pv.Plane(
            center=(sx * 0.5, sy * 0.5, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=sx,
            j_size=sy,
            i_resolution=1,
            j_resolution=1,
        )
        self.plotter.add_mesh(
            bed,
            color=_BED_COLOR,
            opacity=0.18,
            show_edges=False,
            lighting=False,
            name="bed",
            pickable=False,
        )

        self.plotter.add_mesh(
            _make_bed_grid_lines(self.volume.size, step=_BED_GRID_STEP_MM),
            color=_BED_GRID_COLOR,
            line_width=1,
            lighting=False,
            name="bed_grid",
            pickable=False,
        )

        box_edges = pv.Box(bounds=self.volume.bounds).extract_feature_edges()
        self.plotter.add_mesh(
            box_edges,
            color=_VOLUME_EDGE_COLOR,
            line_width=2,
            lighting=False,
            name="volume_box",
            pickable=False,
        )

        axis_len = min(self.volume.size) * 0.12
        for direction, color, name in (
            ((1.0, 0.0, 0.0), "red", "x"),
            ((0.0, 1.0, 0.0), "green", "y"),
            ((0.0, 0.0, 1.0), "deepskyblue", "z"),
        ):
            arrow = pv.Arrow(
                start=(0.0, 0.0, 0.0),
                direction=direction,
                tip_length=0.2,
                tip_radius=0.06,
                shaft_radius=0.025,
                scale=axis_len,
            )
            self.plotter.add_mesh(
                arrow, color=color, lighting=False, name=f"origin_{name}", pickable=False
            )

    def _add_mesh_actor(self) -> None:
        self.mesh_actor = self.plotter.add_mesh(
            self.pv_mesh,
            color=_MESH_COLOR,
            smooth_shading=True,
            specular=0.3,
            specular_power=15,
            name="mesh",
        )

    # ----- voxels -----

    def _rebuild_voxels(self) -> None:
        t0 = time.perf_counter()
        self.voxel_grid = voxelize_solid(self.mesh, pitch=self.pitch)
        self._last_voxelize_ms = (time.perf_counter() - t0) * 1000.0

        image = voxelgrid_to_image_data(self.voxel_grid)
        filled = image.threshold(0.5, scalars="filled").extract_surface(
            algorithm="dataset_surface", progress_bar=False
        )

        if self.voxel_actor is not None:
            self.plotter.remove_actor(self.voxel_actor)
            self.voxel_actor = None

        self.voxel_actor = self.plotter.add_mesh(
            filled,
            color=_VOXEL_COLOR,
            show_edges=False,
            smooth_shading=False,
            specular=0.1,
            specular_power=5,
            name="voxels",
        )
        if self.voxel_actor is not None:
            self.voxel_actor.SetVisibility(self.show_voxels)

    # ----- interactivity -----

    def _set_default_view(self) -> None:
        # Iso-like camera, Z up, framed on the build volume.
        cx, cy, cz = self.volume.center
        diag = float(np.linalg.norm(self.volume.size))
        cam_pos = (cx + diag * 0.9, cy - diag * 1.1, cz + diag * 0.8)
        self.plotter.camera_position = [
            cam_pos,
            (cx, cy, cz * 0.4),  # focal point: slightly below volume center, toward bed
            (0.0, 0.0, 1.0),     # view up: Z
        ]

    def _bind_keys(self) -> None:
        # `r`, `w`, `s` are reserved by VTK's default interactor (reset cam,
        # wireframe, surface). Don't double-bind them — use uppercase / safer
        # alternatives for things we control.
        bindings = {
            "m": ("toggle_mesh", self.toggle_mesh),
            "v": ("toggle_voxels", self.toggle_voxels),
            "b": ("rebuild_voxels", self.rebuild_voxels),
            "bracketleft": ("decrease_pitch", self.decrease_pitch),
            "bracketright": ("increase_pitch", self.increase_pitch),
            "Up": ("increase_pitch", self.increase_pitch),
            "Down": ("decrease_pitch", self.decrease_pitch),
            "F5": ("reset_view", self._reset_view),
        }
        for key, (label, fn) in bindings.items():
            self.plotter.add_key_event(key, _safe(label, fn))

    def _reset_view(self) -> None:
        self._set_default_view()
        self.plotter.render()

    def toggle_mesh(self) -> None:
        self.show_mesh = not self.show_mesh
        if self.mesh_actor is not None:
            self.mesh_actor.SetVisibility(self.show_mesh)
        self._refresh_hud()
        self.plotter.render()

    def toggle_voxels(self) -> None:
        if self.voxel_grid is None:
            self.show_voxels = True
            self.rebuild_voxels()
            return
        self.show_voxels = not self.show_voxels
        if self.voxel_actor is not None:
            self.voxel_actor.SetVisibility(self.show_voxels)
        self._refresh_hud()
        self.plotter.render()

    def rebuild_voxels(self) -> None:
        self.show_voxels = True
        self._rebuild_voxels()
        self._refresh_hud()
        self.plotter.render()

    def increase_pitch(self) -> None:
        self._set_pitch(self.pitch * 1.25)

    def decrease_pitch(self) -> None:
        self._set_pitch(self.pitch / 1.25)

    def _set_pitch(self, value: float) -> None:
        self.pitch = float(max(0.1, min(10.0, value)))
        self._refresh_hud()
        self.plotter.render()

    # ----- HUD -----

    def _init_hud(self) -> None:
        """Create the persistent HUD actor once. Subsequent updates poke its
        text via SetInput() — never recreate actors from inside a key event,
        that's the route to access-violation crashes in the VTK event loop."""
        self._hud_actor = self.plotter.add_text(
            "",
            position="upper_left",
            name="hud",
            font_size=10,
            color=_HUD_COLOR,
            font="courier",
            shadow=True,
        )

    def _hud_text(self) -> str:
        sx, sy, sz = self.volume.size
        lines = [
            "VFF Slicer — Visualizer",
            f"Build volume : {sx:.0f} x {sy:.0f} x {sz:.0f} mm",
            f"Triangles    : {len(self.mesh.faces):,}",
            f"Pitch        : {self.pitch:.3f} mm",
        ]
        if self.voxel_grid is not None:
            nx, ny, nz = self.voxel_grid.shape
            lines.append(
                f"Voxel grid   : {nx} x {ny} x {nz}  "
                f"(filled {self.voxel_grid.filled_count:,})"
            )
            if self._last_voxelize_ms is not None:
                lines.append(f"Voxelize     : {self._last_voxelize_ms:.0f} ms")
        lines.append("")
        lines.append(
            f"mesh: {'on' if self.show_mesh else 'off'}   "
            f"voxels: {'on' if self.show_voxels else 'off'}"
        )
        lines.append("")
        lines.append("[M] mesh  [V] voxels  [B] re-voxelize")
        lines.append("[ [ / ] ] or Up/Down pitch -/+   [F5] reset view")
        return "\n".join(lines)

    def _refresh_hud(self) -> None:
        if self._hud_actor is None:
            return
        text = self._hud_text()
        # CornerAnnotation: SetText(corner, str). 2 = UpperLeft.
        # Both calls are thin VTK setters — safe from inside an event callback,
        # unlike add_text() which removes/re-adds an actor.
        if hasattr(self._hud_actor, "SetText"):
            self._hud_actor.SetText(2, text)
        else:
            self._hud_actor.SetInput(text)

    # ----- run -----

    def show(self) -> None:
        self.plotter.show()
