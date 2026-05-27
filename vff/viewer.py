"""Interactive PyVista visualizer for VFF Slicer.

Scene composition (all in world coords, Z up):
- Build plate at Z=0
- Bed grid (10 mm lines by default)
- Build volume wireframe box
- World-origin axis triad
- Loaded mesh (centered on bed)
- Voxel grid (built on demand)
- Growth-step coloured voxels + per-voxel "toward older" vector arrows

Hotkeys:
- M : toggle mesh
- V : toggle plain voxel shell (builds voxels lazily)
- B : (re)build voxels at current pitch
- G : (re)compute spatial growth from the bed
- C : toggle growth step-coloured voxels
- N : toggle growth vector arrows
- [ / ] or Up / Down : pitch -/+ 25%
- F5 : reset view
- W / S / R : VTK default (wireframe / surface / reset camera)

The growth step slider appears at the bottom of the viewport after G.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Callable

import numpy as np
import pyvista as pv
import trimesh
import vtk
from vtk.util import numpy_support as vns

from .build_volume import BuildVolume
from .growth import GrowthResult, compute_growth
from .voxelize import VoxelGrid, voxelize_solid


def _log(msg: str) -> None:
    """Bulletproof stderr write — uses os.write so a C++ access violation
    cannot strand the message in Python's BufferedWriter. The standard
    print/flush path can leave the last lines unwritten when VTK SIGSEGVs."""
    try:
        os.write(2, (msg + "\n").encode("utf-8", errors="replace"))
    except OSError:
        # fd 2 unavailable (very unusual); fall back to whatever stderr is now.
        try:
            sys.__stderr__.write(msg + "\n")
            sys.__stderr__.flush()
        except Exception:
            pass


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


def _threshold_points_between(filt: "vtk.vtkThresholdPoints", lo: float, hi: float) -> None:
    """vtkThresholdPoints.ThresholdBetween is deprecated in VTK 9.6. Prefer
    the new (SetLower/SetUpper/SetThresholdFunction) API when available."""
    if hasattr(filt, "SetLowerThreshold") and hasattr(filt, "SetUpperThreshold"):
        filt.SetLowerThreshold(float(lo))
        filt.SetUpperThreshold(float(hi))
        if hasattr(filt, "SetThresholdFunction") and hasattr(vtk, "vtkThresholdPoints"):
            # Constants on vtkThresholdPoints in 9.6:
            #   THRESHOLD_BETWEEN / THRESHOLD_LOWER / THRESHOLD_UPPER
            cls = vtk.vtkThresholdPoints
            if hasattr(cls, "THRESHOLD_BETWEEN"):
                filt.SetThresholdFunction(cls.THRESHOLD_BETWEEN)
    else:
        filt.ThresholdBetween(float(lo), float(hi))


def _growth_lut(n_steps: int) -> vtk.vtkLookupTable:
    """Blue (bed) -> red (top) hue ramp for growth-step coloring.
    Same LUT is used for voxels and arrows so they stay visually consistent."""
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(0.0, max(float(n_steps - 1), 1.0))
    lut.SetHueRange(0.66, 0.0)
    lut.SetSaturationRange(0.7, 0.9)
    lut.SetValueRange(0.95, 0.95)
    lut.SetNumberOfTableValues(max(n_steps, 2) * 4)
    lut.Build()
    return lut


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

        # Growth-field state.
        self.growth: GrowthResult | None = None
        self._growth_step_actor = None        # vtkActor for step-colored voxels
        self._growth_step_threshold = None    # vtkThreshold (updated by slider)
        self._growth_vec_actor = None         # vtkActor for arrow glyphs
        self._growth_vec_threshold = None     # vtkThresholdPoints (updated by slider)
        self._growth_slider = None
        self._growth_current_step = 0
        self._growth_max_step = 0
        self._last_growth_ms: float | None = None
        self.show_growth_step = True
        self.show_growth_vec = True

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
        # wireframe, surface). Don't double-bind them — use safer alternatives.
        bindings = {
            "m": ("toggle_mesh", self.toggle_mesh),
            "v": ("toggle_voxels", self.toggle_voxels),
            "b": ("rebuild_voxels", self.rebuild_voxels),
            "g": ("compute_growth", self.do_compute_growth),
            "c": ("toggle_growth_step", self.toggle_growth_step),
            "n": ("toggle_growth_vec", self.toggle_growth_vec),
            "bracketleft": ("decrease_pitch", self.decrease_pitch),
            "bracketright": ("increase_pitch", self.increase_pitch),
            "Up": ("increase_pitch", self.increase_pitch),
            "Down": ("decrease_pitch", self.decrease_pitch),
            "F5": ("reset_view", self._reset_view),
        }
        for key, (label, fn) in bindings.items():
            self.plotter.add_key_event(key, _safe(label, fn))

        # Diagnostic: log every key VTK actually sees, regardless of binding.
        # Lets us see e.g. "user pressed F1, got no callback" or "keysym
        # was Cyrillic_..." on a non-US layout.
        iren = self.plotter.iren.interactor

        def _on_key(_obj, _evt):
            try:
                ks = iren.GetKeySym()
                kc = iren.GetKeyCode()
                _log(f"[vff] keypress: keysym={ks!r}  code={kc!r}")
            except Exception:
                _log("[vff] keypress: failed to read keysym")
                _log(traceback.format_exc())

        def _on_click(_obj, _evt):
            _log("[vff] mouse: left button down")

        # Default priority; observers run AFTER VTK's own dispatch so they
        # don't disturb event handling.
        self._diag_observer_tags = [
            iren.AddObserver("KeyPressEvent", _on_key),
            iren.AddObserver("LeftButtonPressEvent", _on_click),
        ]
        _log("[vff] diagnostic observers installed")

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

    # ----- growth -----

    def do_compute_growth(self) -> None:
        _log("[vff] do_compute_growth: entering")
        if self.voxel_grid is None:
            _log("[vff] do_compute_growth: rebuilding voxels first")
            self.rebuild_voxels()
        if self.voxel_grid is None:
            _log("[vff] do_compute_growth: no voxel grid, aborting")
            return

        _log("[vff] do_compute_growth: running BFS")
        t0 = time.perf_counter()
        self.growth = compute_growth(self.voxel_grid, connectivity=6)
        self._last_growth_ms = (time.perf_counter() - t0) * 1000.0
        _log(
            f"[vff] growth: {self.growth.n_steps} steps, "
            f"{self.growth.painted_count} voxels, {self._last_growth_ms:.0f} ms"
        )

        self._growth_max_step = max(self.growth.n_steps - 1, 0)
        self._growth_current_step = self._growth_max_step

        # Plain mesh + voxel shell would just clutter; turn them off when the
        # growth view comes up. User can re-enable with M / V.
        if self.mesh_actor is not None:
            self.show_mesh = False
            self.mesh_actor.SetVisibility(False)
        if self.voxel_actor is not None:
            self.show_voxels = False
            self.voxel_actor.SetVisibility(False)

        _log("[vff] do_compute_growth: building step actor")
        self._build_growth_step_actor()
        _log("[vff] do_compute_growth: building vector actor")
        self._build_growth_vec_actor()
        _log("[vff] do_compute_growth: adding slider")
        self._add_growth_slider()
        _log("[vff] do_compute_growth: refresh hud + render")
        self._refresh_hud()
        self.plotter.render()
        _log("[vff] do_compute_growth: done")

    def toggle_growth_step(self) -> None:
        if self._growth_step_actor is None:
            return
        self.show_growth_step = not self.show_growth_step
        self._growth_step_actor.SetVisibility(self.show_growth_step)
        self._refresh_hud()
        self.plotter.render()

    def toggle_growth_vec(self) -> None:
        if self._growth_vec_actor is None:
            return
        self.show_growth_vec = not self.show_growth_vec
        self._growth_vec_actor.SetVisibility(self.show_growth_vec)
        self._refresh_hud()
        self.plotter.render()

    def _build_growth_step_actor(self) -> None:
        gr = self.growth
        assert gr is not None
        renderer = self.plotter.renderer

        if self._growth_step_actor is not None:
            renderer.RemoveActor(self._growth_step_actor)
            self._growth_step_actor = None
            self._growth_step_threshold = None

        nx, ny, nz = gr.step.shape

        image = vtk.vtkImageData()
        image.SetDimensions(nx + 1, ny + 1, nz + 1)  # cell dims = point dims - 1
        image.SetSpacing(gr.pitch, gr.pitch, gr.pitch)
        image.SetOrigin(float(gr.origin[0]), float(gr.origin[1]), float(gr.origin[2]))

        # VTK cell ordering matches numpy 'F' (X fastest).
        step_flat = gr.step.astype(np.int32).flatten(order="F")
        step_arr = vns.numpy_to_vtk(step_flat, deep=True, array_type=vtk.VTK_INT)
        step_arr.SetName("step")
        image.GetCellData().AddArray(step_arr)
        image.GetCellData().SetActiveScalars("step")

        threshold = vtk.vtkThreshold()
        threshold.SetInputData(image)
        threshold.SetInputArrayToProcess(
            0, 0, 0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
            "step",
        )
        # vtk 9 API.
        threshold.SetLowerThreshold(0.0)
        threshold.SetUpperThreshold(float(self._growth_current_step))
        if hasattr(threshold, "SetThresholdFunction"):
            threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)

        geom = vtk.vtkGeometryFilter()
        geom.SetInputConnection(threshold.GetOutputPort())

        lut = _growth_lut(gr.n_steps)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(geom.GetOutputPort())
        mapper.SetScalarModeToUseCellData()
        mapper.SelectColorArray("step")
        mapper.SetScalarRange(0.0, max(float(gr.n_steps - 1), 1.0))
        mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetVisibility(self.show_growth_step)

        renderer.AddActor(actor)
        self._growth_step_actor = actor
        self._growth_step_threshold = threshold

    def _build_growth_vec_actor(self) -> None:
        gr = self.growth
        assert gr is not None
        renderer = self.plotter.renderer

        if self._growth_vec_actor is not None:
            renderer.RemoveActor(self._growth_vec_actor)
            self._growth_vec_actor = None
            self._growth_vec_threshold = None

        # Vectors live on step>=1 voxels only.
        xs, ys, zs = np.where(gr.step >= 1)
        if xs.size == 0:
            return

        centers = (
            gr.origin
            + (np.column_stack([xs, ys, zs]).astype(np.float64) + 0.5) * gr.pitch
        )
        vecs = gr.vectors[xs, ys, zs].astype(np.float64)
        steps = gr.step[xs, ys, zs].astype(np.int32)
        n = len(xs)

        # Polydata of vertex cells so vtkThresholdPoints sees the points.
        polydata = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(vns.numpy_to_vtk(centers, deep=True, array_type=vtk.VTK_DOUBLE))
        polydata.SetPoints(points)

        vec_arr = vns.numpy_to_vtk(vecs, deep=True, array_type=vtk.VTK_DOUBLE)
        vec_arr.SetName("vector")
        polydata.GetPointData().SetVectors(vec_arr)

        step_arr = vns.numpy_to_vtk(steps, deep=True, array_type=vtk.VTK_INT)
        step_arr.SetName("step")
        polydata.GetPointData().AddArray(step_arr)
        polydata.GetPointData().SetActiveScalars("step")

        verts = vtk.vtkCellArray()
        # One vertex cell per point.
        conn = np.empty(2 * n, dtype=np.int64)
        conn[0::2] = 1
        conn[1::2] = np.arange(n, dtype=np.int64)
        # vtkCellArray.SetCells using vtkIdTypeArray is fast but version-sensitive.
        # Use InsertNextCell loop — n is at most a few hundred thousand, fine.
        for i in range(n):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        polydata.SetVerts(verts)

        threshold = vtk.vtkThresholdPoints()
        threshold.SetInputData(polydata)
        threshold.SetInputArrayToProcess(
            0, 0, 0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
            "step",
        )
        _threshold_points_between(threshold, 1, self._growth_current_step)

        arrow = vtk.vtkArrowSource()
        arrow.SetTipLength(0.32)
        arrow.SetTipRadius(0.13)
        arrow.SetShaftRadius(0.04)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputConnection(threshold.GetOutputPort())
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetScaleFactor(gr.pitch * 0.7)
        glyph.OrientOn()
        glyph.SetColorModeToColorByScalar()

        lut = _growth_lut(gr.n_steps)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("step")
        mapper.SetScalarRange(0.0, max(float(gr.n_steps - 1), 1.0))
        mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetVisibility(self.show_growth_vec)

        renderer.AddActor(actor)
        self._growth_vec_actor = actor
        self._growth_vec_threshold = threshold

    def _add_growth_slider(self) -> None:
        # Replace any previous growth slider (pitch is controlled by hotkeys).
        try:
            self.plotter.clear_slider_widgets()
        except Exception:
            pass
        self._growth_slider = None

        max_step = self._growth_max_step

        def on_slider(value: float) -> None:
            try:
                step = int(round(float(value)))
                step = max(0, min(step, max_step))
                self._growth_current_step = step
                if self._growth_step_threshold is not None:
                    self._growth_step_threshold.SetUpperThreshold(float(step))
                    self._growth_step_threshold.Modified()
                if self._growth_vec_threshold is not None:
                    # Step=0 has zero vector, so always lower-bound at 1.
                    _threshold_points_between(self._growth_vec_threshold, 1, step)
                    self._growth_vec_threshold.Modified()
                self._refresh_hud()
                self.plotter.render()
            except BaseException:
                _log("[vff] ERROR in growth slider callback:")
                _log(traceback.format_exc())

        self._growth_slider = self.plotter.add_slider_widget(
            callback=on_slider,
            rng=[0, max_step] if max_step > 0 else [0, 1],
            value=max_step,
            title="growth step",
            pointa=(0.30, 0.05),
            pointb=(0.95, 0.05),
            style="modern",
            fmt="%.0f",
        )

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
        if self.growth is not None:
            lines.append(
                f"Growth       : {self.growth.n_steps} steps  "
                f"(showing 0..{self._growth_current_step})"
            )
            if self._last_growth_ms is not None:
                lines.append(f"Growth time  : {self._last_growth_ms:.0f} ms")
        lines.append("")
        layers = (
            f"mesh: {'on' if self.show_mesh else 'off'}   "
            f"voxels: {'on' if self.show_voxels else 'off'}"
        )
        if self.growth is not None:
            layers += (
                f"   growth: {'on' if self.show_growth_step else 'off'}"
                f"   vectors: {'on' if self.show_growth_vec else 'off'}"
            )
        lines.append(layers)
        lines.append("")
        lines.append("[M] mesh  [V] voxels  [B] re-voxel  [G] growth")
        if self.growth is not None:
            lines.append("[C] growth voxels  [N] growth vectors  slider: step")
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
