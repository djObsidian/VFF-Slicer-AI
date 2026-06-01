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
from .deform import deform_mesh, smoothed_depth_field
from .growth import GrowthResult, compute_growth
from .section import build_growth_surfaces
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
        max_tilt_deg: float = 30.0,
        smooth_sigma: float = 2.0,
        depth_method: str = "vectors",
        dz_per_layer: float | None = None,
    ) -> None:
        self.mesh = mesh
        self.volume = volume
        self.pitch = float(initial_pitch)
        # Print-head tilt limit. Set at startup; per the spec, NOT changed
        # dynamically — clamp is part of the underlying growth field.
        self.max_tilt_deg = float(max_tilt_deg)
        # Depth-field parameters — tunable at startup via CLI.
        self.smooth_sigma = float(smooth_sigma)
        self.depth_method = str(depth_method)
        # Z map factor for the exported deformed mesh. None → deform_mesh uses
        # the voxel pitch (1 layer = 1 mm). Must match the dz used by the
        # inverse G-code transform, or the inverse won't undo this deformation.
        self.dz_per_layer = dz_per_layer

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
        # Depth peeling: correct alpha blending when arrows are seen *through*
        # translucent voxel cubes. Without it the back faces of the voxels
        # punch through the arrows. Wrapped because it can fail on some
        # drivers / older OpenGL contexts.
        try:
            self.plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
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
        self._growth_surface_actor = None     # vtkActor for current-step iso-surface
        self._growth_surface_contour = None   # vtkContourFilter (slider sets iso value)
        self._growth_slider = None
        self._growth_current_step = 0
        self._growth_max_step = 0
        self._last_growth_ms: float | None = None
        self.show_growth_step = True
        self.show_growth_vec = True
        self.show_growth_surface = False  # off by default — it obstructs the voxel/arrow view
        # When True, growth-step voxels render fully opaque; when False, they
        # render translucent so the vector field underneath is visible.
        # V hotkey toggles this in growth mode.
        self._growth_step_opaque = True
        self._growth_translucent_alpha = 0.25

        # Deformed (flattened) mesh state. Computed lazily on first D press.
        self.deformed_mesh: trimesh.Trimesh | None = None
        self._deformed_actor = None
        self.show_deformed = False

        # G-code preview state (overlay of extrusion + travel polylines).
        self._gcode_ext_actor = None
        self._gcode_trv_actor = None
        self.show_gcode = False

        # Interactive vertical (∥Z) cross-section state. Built lazily on X.
        self._section_actor = None        # vtkActor for the cut lines
        self._section_overlay_renderer = None  # layer-1 renderer: draw lines on top
        self._section_plane = None        # vtkPlane (vertical: normal in XY)
        self._section_cutter = None       # vtkCutter (slider updates plane)
        self._section_src = None          # in-model-only growth surfaces to cut
        self._section_angle_slider = None
        self._section_pos_slider = None
        self._section_angle_deg = 0.0     # plane normal angle in XY, from +X
        self._section_pos = 0.0           # signed offset along the normal (mm)
        self._section_cx = 0.0
        self._section_cy = 0.0
        self._section_cz = 0.0
        self._section_R = 1.0             # half XY-diagonal — position slider range
        self.show_section = False
        self._screenshot_idx = 0

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
            "h": ("toggle_growth_surface", self.toggle_growth_surface),
            "d": ("toggle_deformed", self.toggle_deformed),
            "o": ("export_deformed", self.export_deformed_default),
            "p": ("toggle_gcode", self.toggle_gcode_preview),
            "x": ("toggle_section", self.toggle_section),
            "i": ("save_section_screenshot", self.save_section_screenshot),
            "j": ("view_section_face_on", self.view_section_face_on),
            "bracketleft": ("decrease_pitch", self.decrease_pitch),
            "bracketright": ("increase_pitch", self.increase_pitch),
            "Up": ("increase_pitch", self.increase_pitch),
            "Down": ("decrease_pitch", self.decrease_pitch),
            "F5": ("reset_view", self._reset_view),
        }
        for key, (label, fn) in bindings.items():
            self.plotter.add_key_event(key, _safe(label, fn))

        # --- Non-US keyboard-layout crash guard -------------------------------
        # On Windows a non-US layout (e.g. Russian) feeds VTK's key translator a
        # keysym/char its default dispatch mishandles → a native access violation
        # deep in the C++ event loop, BEFORE any normal-priority Python observer
        # runs (that's exactly why the old diagnostic never managed to print the
        # offending key). The only Python-reachable lever is a HIGH-priority
        # observer that runs ahead of VTK's interactor style and ABORTS the event
        # for any key that isn't a plain ASCII character or named key. Every
        # binding we use — plus VTK's own w/s/r defaults — is ASCII, so nothing
        # functional is lost; only the untranslatable Cyrillic / dead keys that
        # would otherwise crash get dropped.
        iren = self.plotter.iren.interactor
        guard_tags: list[int] = []

        def _is_safe_key(sym: str, code: str) -> bool:
            if code and code != "\x00":
                return ord(code[0]) < 128           # character key: ASCII only
            return bool(sym) and sym.isascii()       # named key: ASCII keysym

        def _key_guard(caller, _evt):
            try:
                sym = caller.GetKeySym() or ""
            except Exception:
                sym = ""
            try:
                code = caller.GetKeyCode() or ""
            except Exception:
                code = ""
            if not _is_safe_key(sym, code):
                _log(f"[vff] keypress: SWALLOWED non-ASCII key (sym={sym!r} "
                     f"code={code!r}) — non-US layout crash guard")
                # AbortFlagOn on the currently-invoked command stops VTK from
                # calling the lower-priority interactor-style handler (and
                # PyVista's keysym dispatch) for this event — i.e. the code that
                # segfaults never sees the bad key.
                for t in guard_tags:
                    cmd = caller.GetCommand(t)
                    if cmd is not None:
                        cmd.AbortFlagOn()
                return
            _log(f"[vff] keypress: sym={sym!r} code={code!r}")

        def _on_click(_obj, _evt):
            _log("[vff] mouse: left button down")

        # Priority 10 (> the interactor style's default 0) so the guard runs
        # first and its abort takes effect before VTK's crashing dispatch.
        guard_tags.append(iren.AddObserver("KeyPressEvent", _key_guard, 10.0))
        guard_tags.append(iren.AddObserver("CharEvent", _key_guard, 10.0))
        self._diag_observer_tags = guard_tags + [
            iren.AddObserver("LeftButtonPressEvent", _on_click),
        ]
        _log("[vff] key-guard + diagnostic observers installed")

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
        # In growth view, the orange voxel shell is hidden and V is repurposed
        # to toggle the step-coloured voxels between opaque and translucent —
        # the arrow field lives "inside" the voxels, so translucency is the
        # only way to inspect both at once.
        if self._growth_step_actor is not None and self.show_growth_step:
            self._growth_step_opaque = not self._growth_step_opaque
            opacity = 1.0 if self._growth_step_opaque else self._growth_translucent_alpha
            self._growth_step_actor.GetProperty().SetOpacity(opacity)
            self._refresh_hud()
            self.plotter.render()
            return

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
        # Rebuilding voxels invalidates any growth result built from them.
        # Drop growth state so the next G press recomputes cleanly.
        if self.growth is not None:
            _log("[vff] rebuild_voxels: invalidating prior growth state")
            self._invalidate_growth()
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
        """G hotkey: toggle in/out of growth view.

        - First press (no growth yet): voxelize if needed, run BFS, build
          actors + slider, enter growth view with sensible defaults
          (translucent voxels + arrows + surface visible, mesh hidden).
        - Subsequent press: toggles the whole growth-view subsystem on/off.
          The growth result and actors are kept; only visibility flips.
        - Recompute is on-demand: clear with `B` (rebuild voxels) and press
          `G` again, or call `force_recompute_growth` programmatically.
        """
        _log("[vff] do_compute_growth: entering")

        if self.growth is not None:
            # Toggle existing view.
            in_view = (
                self.show_growth_step
                or self.show_growth_vec
                or self.show_growth_surface
            )
            if in_view:
                _log("[vff] do_compute_growth: exiting growth view")
                self._exit_growth_view()
            else:
                _log("[vff] do_compute_growth: re-entering growth view")
                self._enter_growth_view()
            self._refresh_hud()
            self.plotter.render()
            return

        # First time: compute everything.
        if self.voxel_grid is None:
            _log("[vff] do_compute_growth: rebuilding voxels first")
            self.rebuild_voxels()
        if self.voxel_grid is None:
            _log("[vff] do_compute_growth: no voxel grid, aborting")
            return

        _log("[vff] do_compute_growth: running BFS")
        t0 = time.perf_counter()
        self.growth = compute_growth(
            self.voxel_grid,
            connectivity=26,
            max_tilt_deg=self.max_tilt_deg,
        )
        self._last_growth_ms = (time.perf_counter() - t0) * 1000.0
        _log(
            f"[vff] growth: {self.growth.n_steps} steps, "
            f"{self.growth.painted_count} voxels, {self._last_growth_ms:.0f} ms"
        )

        self._growth_max_step = max(self.growth.n_steps - 1, 0)
        self._growth_current_step = self._growth_max_step

        _log("[vff] do_compute_growth: building step actor")
        self._build_growth_step_actor()
        _log("[vff] do_compute_growth: building vector actor")
        self._build_growth_vec_actor()
        _log("[vff] do_compute_growth: building surface actor")
        self._build_growth_surface_actor()
        _log("[vff] do_compute_growth: adding slider")
        self._add_growth_slider()
        _log("[vff] do_compute_growth: entering growth view")
        self._enter_growth_view()
        self._refresh_hud()
        self.plotter.render()
        _log("[vff] do_compute_growth: done")

    def _enter_growth_view(self) -> None:
        """Show growth visualisation: translucent voxels (so arrows + surface
        underneath are visible), arrows, and surface. Hide raw mesh, voxel
        shell, and deformed mesh — only one view at a time."""
        self.show_mesh = False
        self.show_voxels = False
        self.show_deformed = False
        self.show_growth_step = True
        self._growth_step_opaque = False  # translucent so vectors + surface show
        self.show_growth_vec = True
        self.show_growth_surface = True
        self._apply_growth_visibility()

    def _exit_growth_view(self) -> None:
        """Hide all growth actors, bring the mesh back."""
        self.show_growth_step = False
        self.show_growth_vec = False
        self.show_growth_surface = False
        self.show_deformed = False
        self.show_mesh = True
        self._apply_growth_visibility()

    def _apply_growth_visibility(self) -> None:
        if self.mesh_actor is not None:
            self.mesh_actor.SetVisibility(self.show_mesh)
        if self.voxel_actor is not None:
            self.voxel_actor.SetVisibility(self.show_voxels)
        if self._growth_step_actor is not None:
            self._growth_step_actor.SetVisibility(self.show_growth_step)
            self._growth_step_actor.GetProperty().SetOpacity(
                1.0 if self._growth_step_opaque else self._growth_translucent_alpha
            )
        if self._growth_vec_actor is not None:
            self._growth_vec_actor.SetVisibility(self.show_growth_vec)
        if self._growth_surface_actor is not None:
            self._growth_surface_actor.SetVisibility(self.show_growth_surface)
        if self._deformed_actor is not None:
            self._deformed_actor.SetVisibility(self.show_deformed)

    def _invalidate_growth(self) -> None:
        """Drop all growth state and actors. Next G press recomputes from
        scratch. Used when the underlying voxel grid changes (B hotkey)."""
        renderer = self.plotter.renderer
        for attr in (
            "_growth_step_actor",
            "_growth_vec_actor",
            "_growth_surface_actor",
            "_deformed_actor",
            "_section_actor",
        ):
            actor = getattr(self, attr, None)
            if actor is not None:
                renderer.RemoveActor(actor)
                setattr(self, attr, None)
        self._growth_step_threshold = None
        self._growth_vec_threshold = None
        self._growth_surface_contour = None
        if self._growth_slider is not None:
            try:
                self.plotter.clear_slider_widgets()
            except Exception:
                pass
            self._growth_slider = None
        # clear_slider_widgets() above also drops the section sliders.
        self._section_plane = None
        self._section_cutter = None
        self._section_src = None
        self._section_angle_slider = None
        self._section_pos_slider = None
        self.show_section = False
        if self._section_overlay_renderer is not None:
            try:
                rw = getattr(self.plotter, "render_window", None) or self.plotter.ren_win
                rw.RemoveRenderer(self._section_overlay_renderer)
            except Exception:
                pass
            self._section_overlay_renderer = None
        self.growth = None
        self.deformed_mesh = None
        self.show_growth_step = False
        self.show_growth_vec = False
        self.show_growth_surface = False
        self.show_deformed = False
        # Bring the mesh back into the picture.
        self.show_mesh = True
        if self.mesh_actor is not None:
            self.mesh_actor.SetVisibility(True)

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

    def toggle_growth_surface(self) -> None:
        if self._growth_surface_actor is None:
            return
        self.show_growth_surface = not self.show_growth_surface
        self._growth_surface_actor.SetVisibility(self.show_growth_surface)
        self._refresh_hud()
        self.plotter.render()

    def _ensure_deformed_mesh(self) -> bool:
        """Compute the deformed mesh on demand. Returns True if available."""
        if self.deformed_mesh is not None:
            return True
        if self.growth is None:
            _log("[vff] _ensure_deformed_mesh: growth not computed; computing now")
            self.do_compute_growth()
            if self.growth is None:
                return False
        _log("[vff] _ensure_deformed_mesh: deforming")
        t0 = time.perf_counter()
        self.deformed_mesh = deform_mesh(
            self.mesh, self.growth,
            dz_per_layer=self.dz_per_layer,
            smooth_sigma=self.smooth_sigma,
            depth_method=self.depth_method,
        )
        _log(
            f"[vff] deform: Z [{self.deformed_mesh.bounds[0, 2]:.2f}, "
            f"{self.deformed_mesh.bounds[1, 2]:.2f}] mm  "
            f"({(time.perf_counter() - t0) * 1000.0:.0f} ms)"
        )
        return True

    def save_deformed(self, path: str) -> bool:
        """Compute (if needed) and export the deformed mesh to `path`. Any
        format trimesh.export supports works from the extension: .stl, .ply,
        .obj, .glb, .gltf, .dae, .off. Reports volume vs original for sanity."""
        if not self._ensure_deformed_mesh():
            _log(f"[vff] save_deformed: could not build deformed mesh, skipping export to {path}")
            return False
        try:
            self.deformed_mesh.export(path)
        except Exception as e:
            _log(f"[vff] save_deformed: export failed: {e!r}")
            return False
        v_orig = float(self.mesh.volume)
        v_def = float(self.deformed_mesh.volume)
        ratio = (v_def / v_orig) if v_orig > 0 else 0.0
        _log(
            f"[vff] saved deformed mesh -> {path}  "
            f"({len(self.deformed_mesh.faces):,} faces, "
            f"vol_orig={v_orig:.0f} mm^3, vol_def={v_def:.0f} mm^3, ratio={ratio:.3f})"
        )
        return True

    def export_deformed_default(self) -> None:
        """O hotkey: save deformed mesh to a default name in CWD."""
        from pathlib import Path
        default_path = Path.cwd() / "deformed_mesh.stl"
        self.save_deformed(str(default_path))

    def load_gcode_preview(self, path: str) -> bool:
        """Parse a G-code file and overlay its extrusion + travel polylines
        in the viewer. PrusaSlicer's own preview tops out at planar layers;
        ours just draws whatever lines the file specifies."""
        from .gcode_preview import parse_gcode
        import time as _time
        t0 = _time.perf_counter()
        try:
            data = parse_gcode(path)
        except Exception as e:
            _log(f"[vff] load_gcode_preview: parse failed: {e!r}")
            return False
        dt = (_time.perf_counter() - t0) * 1000.0
        _log(
            f"[vff] gcode '{path}': {data['n_extrusion_moves']:,} extrusion + "
            f"{data['n_travel_moves']:,} travel moves  ({dt:.0f} ms parse)"
        )
        renderer = self.plotter.renderer

        # Remove old actors if any.
        for attr in ("_gcode_ext_actor", "_gcode_trv_actor"):
            actor = getattr(self, attr, None)
            if actor is not None:
                renderer.RemoveActor(actor)
                setattr(self, attr, None)

        if data["extrusion_points"].shape[0] > 0:
            pd = pv.PolyData(data["extrusion_points"], lines=data["extrusion_lines"])
            # One scalar per VTK cell (line segment) — Z of segment midpoint.
            pd.cell_data["z_mid"] = data["extrusion_step"]
            self._gcode_ext_actor = self.plotter.add_mesh(
                pd, scalars="z_mid", cmap="plasma",
                line_width=2, show_scalar_bar=False, lighting=False,
                name="gcode_extrusion",
            )
        if data["travel_points"].shape[0] > 0:
            pd_t = pv.PolyData(data["travel_points"], lines=data["travel_lines"])
            self._gcode_trv_actor = self.plotter.add_mesh(
                pd_t, color="#666666",
                line_width=1, lighting=False, opacity=0.4,
                name="gcode_travel",
            )
            if self._gcode_trv_actor is not None:
                # Hidden by default — travels clutter the picture.
                self._gcode_trv_actor.SetVisibility(False)

        # Start visible.
        self.show_gcode = True
        if self._gcode_ext_actor is not None:
            self._gcode_ext_actor.SetVisibility(True)
        self._refresh_hud()
        self.plotter.render()
        return True

    def toggle_gcode_preview(self) -> None:
        """P hotkey: toggle the G-code overlay actors on/off."""
        if self._gcode_ext_actor is None:
            _log("[vff] toggle_gcode_preview: no G-code loaded yet (use --preview-gcode at startup)")
            return
        self.show_gcode = not self.show_gcode
        if self._gcode_ext_actor is not None:
            self._gcode_ext_actor.SetVisibility(self.show_gcode)
        if self._gcode_trv_actor is not None:
            # Travels follow the master toggle but stay hidden when overall is on
            # (user can use a separate toggle if they want travels).
            self._gcode_trv_actor.SetVisibility(False)
        self._refresh_hud()
        self.plotter.render()

    def toggle_deformed(self) -> None:
        """D: enter the 'flattened' view — show the deformed mesh whose Z is
        proportional to growth step. Growth iso-surfaces become parallel XY
        planes. Press again to return to the natural-space view."""
        if not self._ensure_deformed_mesh():
            return

        if self._deformed_actor is None:
            self._deformed_actor = self.plotter.add_mesh(
                trimesh_to_pv(self.deformed_mesh),
                color="#d850c0",  # magenta — clearly different from anything else
                smooth_shading=True,
                specular=0.3,
                specular_power=12,
                opacity=0.95,
                name="deformed_mesh",
            )
            if self._deformed_actor is not None:
                self._deformed_actor.SetVisibility(False)

        self.show_deformed = not self.show_deformed
        # In flattened view, hide the natural-space stuff so it's not confusing.
        actors_to_dim = [
            self.mesh_actor,
            self.voxel_actor,
            self._growth_step_actor,
            self._growth_vec_actor,
            self._growth_surface_actor,
        ]
        if self.show_deformed:
            for a in actors_to_dim:
                if a is not None:
                    a.SetVisibility(False)
            if self._deformed_actor is not None:
                self._deformed_actor.SetVisibility(True)
        else:
            # Restore last visibility states.
            if self.mesh_actor is not None:
                self.mesh_actor.SetVisibility(self.show_mesh)
            if self.voxel_actor is not None:
                self.voxel_actor.SetVisibility(self.show_voxels)
            if self._growth_step_actor is not None:
                self._growth_step_actor.SetVisibility(self.show_growth_step)
            if self._growth_vec_actor is not None:
                self._growth_vec_actor.SetVisibility(self.show_growth_vec)
            if self._growth_surface_actor is not None:
                self._growth_surface_actor.SetVisibility(self.show_growth_surface)
            if self._deformed_actor is not None:
                self._deformed_actor.SetVisibility(False)
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

        # 2D-style flat arrow glyphs, like ParaView / Ansys vector overlays.
        # vtkGlyphSource2D::ThickArrow is a filled 2D arrow in the XY plane;
        # vtkGlyph3D rotates it to align with each vector. The flat look reads
        # well in dense fields and doesn't fight the surface for attention.
        arrow = vtk.vtkGlyphSource2D()
        arrow.SetGlyphTypeToThickArrow()
        arrow.SetScale(1.0)
        arrow.FilledOn()
        arrow.SetCenter(0.5, 0.0, 0.0)  # base at origin, tip at +X

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputConnection(threshold.GetOutputPort())
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()  # arrow length = |vector| * SetScaleFactor
        glyph.SetScaleFactor(gr.pitch * 0.85)  # √3 * 0.85 ≈ 1.47 voxel — visible
        glyph.OrientOn()
        glyph.ScalingOn()
        glyph.SetColorModeToColorByScalar()

        lut = _growth_lut(gr.n_steps)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOff()  # solid red — user wanted ParaView/Ansys look

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.92, 0.18, 0.18)  # red
        actor.GetProperty().SetLighting(False)          # uniform colour, no shading
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        actor.SetVisibility(self.show_growth_vec)

        renderer.AddActor(actor)
        self._growth_vec_actor = actor
        self._growth_vec_threshold = threshold

    def _build_growth_surface_actor(self) -> None:
        """Growth (layer) surfaces: iso-surfaces of the depth field, ONE per
        growth step, all of them from 0 up to the current slider step.

        With depth_method='vectors' the depth field is the potential whose
        gradient is the clamped growth vector field, so each iso-surface is
        perpendicular to the growth direction — these ARE the non-planar
        slicer's target layer surfaces. Coloured blue(bed)->red(top) on the
        same LUT as the voxels. Pipeline is persistent; the slider only
        changes how many contours are active.

        The depth field is rescaled so its in-model max equals n_steps-1,
        i.e. it lines up with the BFS-step scale the slider is numbered in.
        Uniform scaling doesn't move the level sets, only relabels them, so
        the perpendicular-to-vectors property is preserved.
        """
        gr = self.growth
        assert gr is not None
        renderer = self.plotter.renderer

        if self._growth_surface_actor is not None:
            renderer.RemoveActor(self._growth_surface_actor)
            self._growth_surface_actor = None
            self._growth_surface_contour = None

        nx, ny, nz = gr.step.shape

        image = vtk.vtkImageData()
        image.SetDimensions(nx + 1, ny + 1, nz + 1)
        image.SetSpacing(gr.pitch, gr.pitch, gr.pitch)
        image.SetOrigin(float(gr.origin[0]), float(gr.origin[1]), float(gr.origin[2]))

        # Smoothed continuous depth field — see deform.smoothed_depth_field.
        # Inside the model: BFS step values, Gaussian-smoothed so symmetric
        # features (e.g. propeller blades) end up with symmetric depth even
        # though raw BFS is anisotropic in voxel-grid coordinates.
        # Outside the model: vertical depth (k - k_bed_layer), so iso-surfaces
        # extend as horizontal planes with vertical normals — no flood-fill
        # perturbations near the model boundary.
        # Surface viz uses outside_mode='vertical' so iso-surfaces in air are
        # horizontal planes (vertical normals). Deformation in deform_mesh
        # uses the default outside_mode='extend' for boundary continuity.
        extended = smoothed_depth_field(
            gr, sigma=self.smooth_sigma, method=self.depth_method,
            outside_mode="vertical",
        )
        # Rescale so the in-model max matches the BFS-step scale (slider range
        # is 0..n_steps-1). The 'vectors' potential has its own units (it
        # integrates vector magnitudes, not integer steps), so without this
        # the top of the model would sit beyond the last slider position and
        # never get a surface. Geometry of the level sets is unchanged.
        inside = gr.step >= 0
        if inside.any() and gr.n_steps > 1:
            fmax = float(np.nanmax(extended[inside]))
            if fmax > 1e-6:
                extended = extended * (float(gr.n_steps - 1) / fmax)
        step_flat = extended.flatten(order="F")
        arr = vns.numpy_to_vtk(step_flat, deep=True, array_type=vtk.VTK_FLOAT)
        arr.SetName("step")
        image.GetCellData().AddArray(arr)
        image.GetCellData().SetActiveScalars("step")

        # Marching cubes needs point data. Convert cell -> point (each vertex
        # gets the average of its incident cells' values).
        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputData(image)
        c2p.PassCellDataOff()

        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(c2p.GetOutputPort())
        contour.SetInputArrayToProcess(
            0, 0, 0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
            "step",
        )
        _n_surf = self._growth_current_step + 1
        contour.SetNumberOfContours(_n_surf)
        for _i in range(_n_surf):
            contour.SetValue(_i, float(_i) + 0.5)
        contour.ComputeNormalsOn()

        # Smooth the iso-surface — flood-fill creates step-shaped artifacts
        # near the model boundary; sinc smoothing flattens them without
        # eroding the geometry the way Laplacian smoothing does.
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(20)
        smoother.SetPassBand(0.05)
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.SetFeatureAngle(60)
        normals.ConsistencyOn()
        normals.SplittingOff()

        lut = _growth_lut(gr.n_steps)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOn()
        mapper.SetScalarRange(0.0, max(float(gr.n_steps - 1), 1.0))
        mapper.SetLookupTable(lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.18)
        actor.GetProperty().SetAmbient(0.30)
        actor.GetProperty().SetDiffuse(0.70)
        actor.GetProperty().SetSpecular(0.10)
        actor.GetProperty().SetInterpolationToGouraud()
        actor.SetVisibility(self.show_growth_surface)

        renderer.AddActor(actor)
        self._growth_surface_actor = actor
        self._growth_surface_contour = contour

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
                if self._growth_surface_contour is not None:
                    _c = self._growth_surface_contour
                    _n = step + 1
                    _c.SetNumberOfContours(_n)
                    for _i in range(_n):
                        _c.SetValue(_i, float(_i) + 0.5)
                    _c.Modified()
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

    # ----- interactive Z-parallel cross-section -----

    def _ensure_section_built(self) -> bool:
        """Build the cross-section pipeline once: in-model-only growth surfaces
        (all steps) + a vertical cutting plane + cutter + line actor. Returns
        True if ready. Requires growth — computes it on demand."""
        if self._section_actor is not None:
            return True
        if self.growth is None:
            _log("[vff] _ensure_section_built: computing growth first")
            self.do_compute_growth()
            if self.growth is None:
                return False
        _log("[vff] _ensure_section_built: building in-model surfaces to cut")
        surf = build_growth_surfaces(
            self.growth, smooth_sigma=self.smooth_sigma, depth_method=self.depth_method,
        )
        self._section_src = surf

        b = self.mesh.bounds
        self._section_cx = 0.5 * float(b[0, 0] + b[1, 0])
        self._section_cy = 0.5 * float(b[0, 1] + b[1, 1])
        self._section_cz = 0.5 * float(b[0, 2] + b[1, 2])
        self._section_R = 0.5 * float(np.hypot(b[1, 0] - b[0, 0], b[1, 1] - b[0, 1]))

        # Vertical plane: normal lies in XY (z-component 0), so the plane always
        # contains the Z direction. The cutter slices the layer surfaces along it.
        plane = vtk.vtkPlane()
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(surf)

        lut = _growth_lut(self.growth.n_steps)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cutter.GetOutputPort())
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("step")
        mapper.SetScalarRange(0.0, max(float(self.growth.n_steps - 1), 1.0))
        mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(3)
        actor.GetProperty().SetLighting(False)
        actor.SetVisibility(False)

        # Draw the section lines ON TOP of everything. They live INSIDE the
        # model volume, so in the main renderer the ghosted mesh/voxels in
        # front occlude them — only the near/centre part shows ("только в
        # центре"). A layer-1 renderer that shares the camera and clears the
        # depth buffer makes them overlay the whole scene at any angle/pos.
        rw = getattr(self.plotter, "render_window", None) or self.plotter.ren_win
        overlay = None
        try:
            if rw.GetNumberOfLayers() < 2:
                rw.SetNumberOfLayers(2)
            overlay = vtk.vtkRenderer()
            overlay.SetLayer(1)
            overlay.InteractiveOff()
            overlay.SetPreserveColorBuffer(1)   # keep the base image underneath
            # PreserveDepthBuffer stays 0 -> depth cleared -> lines always on top.
            overlay.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
            overlay.AddActor(actor)
            rw.AddRenderer(overlay)
        except Exception:
            _log("[vff] section overlay renderer failed; using main renderer")
            _log(traceback.format_exc())
            overlay = None
            self.plotter.renderer.AddActor(actor)
        self._section_overlay_renderer = overlay

        self._section_plane = plane
        self._section_cutter = cutter
        self._section_actor = actor
        self._update_section_plane()
        self._add_section_sliders()
        return True

    def _update_section_plane(self) -> None:
        if self._section_plane is None or self._section_cutter is None:
            return
        th = float(np.deg2rad(self._section_angle_deg))
        nx_, ny_ = float(np.cos(th)), float(np.sin(th))
        d = self._section_pos
        self._section_plane.SetNormal(nx_, ny_, 0.0)
        self._section_plane.SetOrigin(
            self._section_cx + d * nx_, self._section_cy + d * ny_, self._section_cz
        )
        self._section_cutter.Modified()

    def _add_section_sliders(self) -> None:
        R = self._section_R

        def on_angle(value: float) -> None:
            try:
                self._section_angle_deg = float(value)
                self._update_section_plane()
                self._view_section_face_on(reset=False)  # keep facing the plane
                self._refresh_hud()
                self.plotter.render()
            except BaseException:
                _log("[vff] ERROR in section angle slider:")
                _log(traceback.format_exc())

        def on_pos(value: float) -> None:
            try:
                self._section_pos = float(value)
                self._update_section_plane()
                self._refresh_hud()
                self.plotter.render()
            except BaseException:
                _log("[vff] ERROR in section pos slider:")
                _log(traceback.format_exc())

        self._section_angle_slider = self.plotter.add_slider_widget(
            callback=on_angle, rng=[0.0, 180.0], value=self._section_angle_deg,
            title="section angle", pointa=(0.04, 0.30), pointb=(0.30, 0.30),
            style="modern", fmt="%.0f",
        )
        self._section_pos_slider = self.plotter.add_slider_widget(
            callback=on_pos, rng=[-R, R], value=self._section_pos,
            title="section pos", pointa=(0.04, 0.22), pointb=(0.30, 0.22),
            style="modern", fmt="%.1f",
        )

    def _section_set_sliders_enabled(self, on: bool) -> None:
        for w in (self._section_angle_slider, self._section_pos_slider):
            if w is not None:
                try:
                    w.SetEnabled(1 if on else 0)
                except Exception:
                    pass

    def _apply_section_dim(self, on: bool) -> None:
        """Section on -> ghost everything else so the crisp section lines pop.
        Section off -> restore each actor's normal opacity."""
        dim = 0.07
        step_normal = 1.0 if self._growth_step_opaque else self._growth_translucent_alpha
        pairs = [
            (self.mesh_actor, 1.0),
            (self.voxel_actor, 1.0),
            (self._growth_step_actor, step_normal),
            (self._growth_vec_actor, 1.0),
            (self._growth_surface_actor, 0.18),
            (self._deformed_actor, 0.95),
        ]
        for actor, normal in pairs:
            if actor is not None:
                actor.GetProperty().SetOpacity(dim if on else normal)

    def toggle_section(self) -> None:
        """X: toggle the interactive vertical (∥Z) cross-section. Angle and
        position are driven by the two sliders; everything else is ghosted
        while it's on. Press I to save a screenshot of the result."""
        if not self.show_section:
            if not self._ensure_section_built():
                _log("[vff] toggle_section: no growth, cannot build section")
                return
            self.show_section = True
            if self._section_actor is not None:
                self._section_actor.SetVisibility(True)
            self._apply_section_dim(True)
            self._section_set_sliders_enabled(True)
            # Face the cut straight-on (orthographic) so it fills the view —
            # an oblique view shows only a thin slanted slice in the centre.
            self._view_section_face_on(reset=True)
        else:
            self.show_section = False
            if self._section_actor is not None:
                self._section_actor.SetVisibility(False)
            self._apply_section_dim(False)
            self._section_set_sliders_enabled(False)
            try:
                self.plotter.disable_parallel_projection()
            except Exception:
                pass
        self._refresh_hud()
        self.plotter.render()

    def save_section_screenshot(self) -> None:
        """I: save the current viewer image (ghosted scene + section lines)."""
        from pathlib import Path
        self._screenshot_idx += 1
        path = Path.cwd() / f"section_view_{self._screenshot_idx:03d}.png"
        try:
            self.plotter.screenshot(str(path))
            _log(f"[vff] saved screenshot -> {path}")
        except Exception as e:
            _log(f"[vff] screenshot failed: {e!r}")

    def _view_section_face_on(self, reset: bool = True) -> None:
        """Point the camera straight down the section-plane normal, orthographic,
        so the whole cut fills the view. Without this an oblique camera shows
        only a thin slanted slice — which reads as 'not the whole surface'.

        reset=True refits the zoom (used on enter / J); reset=False just
        re-aims (used while dragging the angle slider, so it doesn't jump)."""
        if not self.show_section or self._section_plane is None:
            return
        th = float(np.deg2rad(self._section_angle_deg))
        nx_, ny_ = float(np.cos(th)), float(np.sin(th))
        focal = (self._section_cx, self._section_cy, self._section_cz)
        dist = 2.5 * max(self._section_R, 1.0)
        pos = (focal[0] + nx_ * dist, focal[1] + ny_ * dist, focal[2])
        try:
            self.plotter.enable_parallel_projection()
        except Exception:
            pass
        self.plotter.camera_position = [pos, focal, (0.0, 0.0, 1.0)]
        if reset:
            try:
                self.plotter.reset_camera()
                # reset_camera frames the whole 250 mm build volume; tighten in
                # to frame the section (which extends ~2x the model footprint).
                self.plotter.camera.zoom(1.3)
            except Exception:
                pass

    def view_section_face_on(self) -> None:
        """J: re-aim the camera face-on to the current section plane."""
        if not self.show_section:
            return
        self._view_section_face_on(reset=True)
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
            f"Max tilt     : {self.max_tilt_deg:.1f} deg from +Z",
            f"Depth field  : {self.depth_method}, sigma={self.smooth_sigma:.2f}",
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
            opacity_tag = "opaque" if self._growth_step_opaque else "translucent"
            layers += (
                f"   growth: {'on' if self.show_growth_step else 'off'} ({opacity_tag})"
                f"   vectors: {'on' if self.show_growth_vec else 'off'}"
                f"   surface: {'on' if self.show_growth_surface else 'off'}"
            )
        lines.append(layers)
        lines.append("")
        lines.append("[M] mesh  [V] voxels  [B] re-voxel  [G] growth")
        if self.growth is not None:
            lines.append("[V] flips growth voxels opaque <-> translucent")
            lines.append("[C] voxels on/off  [N] vectors on/off  [H] surface  slider: step")
            lines.append(f"[D] toggle deformed (flattened) mesh: {'on' if self.show_deformed else 'off'}")
            sec = f"[X] section ∥Z: {'on' if self.show_section else 'off'}"
            if self.show_section:
                sec += f"  angle={self._section_angle_deg:.0f}deg  pos={self._section_pos:.1f}mm  (sliders)"
            sec += "   [I] save  [J] face-on"
            lines.append(sec)
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
