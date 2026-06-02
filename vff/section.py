"""Render an XZ-plane cross-section of the print's layer surfaces to an image.

Headless / off-screen — no interactive window, so it sidesteps the
interactive-viewer issues and is good for batch figures.

What the layers are depends on `deform_mode` (so the section matches the path
that actually deforms the part):
  - "3d": the ACTUAL layers the full-3D map produces — iso-surfaces of the
    deformed-Z field Φ_z over the original grid, clipped to the part. A planar
    slicer cuts the deformed mesh at flat z_def planes; these are the curved
    surfaces those map back to. A FOLD shows as a self-touching/closed curve —
    a faithful debug view of what the 3D method generates.
  - "z-only": iso-surfaces of the scalar depth field (the vector-integrated /
    harmonic potential), what the Z-only path uses.

Cutting with the XZ plane (normal +Y) at the model's Y centre yields the layer
curves of a vertical cross-section. The model's OWN section outline is drawn on
top in a contrasting colour (the part boundary cut by the same plane), always
legible over the layer lines. Nothing else — no voxels or arrows.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import trimesh
from scipy.ndimage import distance_transform_edt

from .deform import smoothed_depth_field
from .growth import GrowthResult, compute_growth
from .voxelize import voxelize_solid


def build_growth_surfaces(
    growth: GrowthResult,
    smooth_sigma: float = 2.0,
    depth_method: str = "vectors",
    margin_factor: float = 2.0,
) -> pv.PolyData:
    """All growth surfaces (one iso-surface per layer) as a single PolyData,
    coloured by step, spanning a full plane AROUND the part — not just its
    outline.

    The layer surfaces are only physically defined INSIDE the solid, where the
    growth vectors live (the depth field's level sets are perpendicular to
    them). Outside, there are no vectors, so we have to choose how to continue
    each surface into the air. Two earlier choices both misbehaved at blade
    tips:

      - "vertical" (depth = height): the inside potential and the vertical
        fill disagree by several layers at the boundary (the growth had to
        climb a long non-vertical path the vertical fill ignores), so each
        iso-surface was forced to DIVE down where a blade ends — the artifact
        the user spotted.
      - "extend" (nearest inside value): no dive, but surfaces fold up into
        steep walls at the tips, inheriting the tip's slope as a flat shelf.

    This uses a FIRST-ORDER (tangent) continuation, chosen by the user: every
    air cell takes the value AND gradient of the nearest in-model cell —
    value + ∇φ·(p − p_nearest). So each surface leaves the solid smoothly and
    continues as the tangent plane of its local patch, then flattens out with
    distance. It's the "fit an approximating plane and blend smoothly" idea,
    done per-patch (locally) rather than one global plane, because a 12-blade
    propeller's layer is a 12-lobed dome — a single global plane would average
    the lobes away and lie at every blade.

    `margin_factor` pads the field outward (XY both sides + some Z) so the
    tangent continuation has room to read as a real plane. 1.0 = grid only;
    2.0 = ~2x footprint.
    """
    gr = growth
    nx, ny, nz = gr.step.shape
    inside = gr.step >= 0

    # Inside-only depth (scaled to the integer step scale). We deliberately
    # IGNORE smoothed_depth_field's outside fill here and recompute the air
    # ourselves with the tangent law below.
    #
    # outside_mode MUST be "extend" (nearest-inside), NOT "vertical": the air
    # fill is overwritten below, but the Gaussian smoothing inside it still SEES
    # those outside values at the model boundary. "vertical" puts height (k −
    # k_bed) in the air, which near a wide overhang (a mushroom cap) is much
    # LOWER than the inside path-distance there — so smoothing drags the
    # cap-edge values down and manufactures a FALSE interior maximum, i.e. a
    # closed iso-surface that the deformation does not actually have. "extend"
    # is continuous across the boundary, so smoothing introduces no such dip;
    # a genuine interior extremum (a real fold) still shows. (Measured on the
    # mushroom: "vertical" → 2 spurious closed loops, "extend" → 0.)
    field_in = smoothed_depth_field(
        gr, sigma=smooth_sigma, method=depth_method, outside_mode="extend"
    )
    scale = 1.0
    if inside.any() and gr.n_steps > 1:
        fmax = float(np.nanmax(field_in[inside]))
        if fmax > 1e-6:
            scale = float(gr.n_steps - 1) / fmax
    phi_in = (field_in * scale).astype(np.float32)

    # Padded grid: known = in-model cells (carry φ); everything else unknown.
    if margin_factor > 1.0:
        px = int(round((margin_factor - 1.0) * 0.5 * nx))
        py = int(round((margin_factor - 1.0) * 0.5 * ny))
    else:
        px = py = 0
    # No upward Z pad: we clip everything above the model's top Z at the end,
    # so layer planes overhead would only be built to be thrown away.
    pz = 0
    NX, NY, NZ = nx + 2 * px, ny + 2 * py, nz + pz

    known = np.zeros((NX, NY, NZ), dtype=bool)
    val = np.full((NX, NY, NZ), np.nan, dtype=np.float32)
    known[px:px + nx, py:py + ny, 0:nz][inside] = True
    val[px:px + nx, py:py + ny, 0:nz][inside] = phi_in[inside]

    unknown = ~known
    if unknown.any() and known.any():
        # Nearest in-model cell for every air cell (Euclidean).
        _, idx = distance_transform_edt(
            unknown, return_distances=True, return_indices=True
        )
        bi, bj, bk = idx
        u = unknown
        nb_i, nb_j, nb_k = bi[u], bj[u], bk[u]

        # Zero-order fill first (nearest value), so the gradient is defined
        # everywhere; then read the gradient at the nearest in-model cell and
        # extrapolate linearly from it — the tangent plane of that patch.
        filled0 = val.copy()
        filled0[u] = val[nb_i, nb_j, nb_k]
        gx, gy, gz = np.gradient(filled0)
        ii, jj, kk = np.indices(val.shape)
        val[u] = (
            val[nb_i, nb_j, nb_k]
            + gx[nb_i, nb_j, nb_k] * (ii[u] - nb_i)
            + gy[nb_i, nb_j, nb_k] * (jj[u] - nb_j)
            + gz[nb_i, nb_j, nb_k] * (kk[u] - nb_k)
        )

    origin = (
        float(gr.origin[0]) - px * gr.pitch,
        float(gr.origin[1]) - py * gr.pitch,
        float(gr.origin[2]),
    )
    grid = pv.ImageData(
        dimensions=(NX + 1, NY + 1, NZ + 1),
        spacing=(gr.pitch, gr.pitch, gr.pitch),
        origin=origin,
    )
    grid.cell_data["step"] = val.flatten(order="F")
    pgrid = grid.cell_data_to_point_data()

    # One iso per physical layer of the part (0..n_steps-1). Each continues as
    # its tangent plane into the air via the fill above.
    isos = [float(i) + 0.5 for i in range(max(gr.n_steps, 1))]
    surf = pgrid.contour(isosurfaces=isos, scalars="step")

    # Drop everything ABOVE the model. The layer surfaces are only meaningful
    # up to where the part ends; the tangent continuation overhead is just
    # empty air-plane clutter. Clip at the top face of the highest occupied
    # voxel layer (invert=True keeps the side below +Z).
    if inside.any() and surf.n_points > 0:
        k_top = int(np.max(np.where(inside.any(axis=(0, 1)))))
        z_top = float(gr.origin[2]) + (k_top + 1) * gr.pitch
        surf = surf.clip(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, z_top), invert=True)
    return surf


def build_deform_layer_surfaces(
    dmap, gr: GrowthResult, mesh: trimesh.Trimesh | None,
    *, layer_step: float | None = None,
) -> tuple[pv.PolyData, float, float]:
    """Iso-surfaces of the deformed-Z field Φ_z over the ORIGINAL grid — the
    ACTUAL layers the full-3D map (deform3d) produces, in original coordinates.

    A planar slicer cuts the deformed mesh at flat planes z_def = c; the inverse
    maps those back to the curved surface {Φ_z(p) = c} in the part. So these are
    exactly the layers the 3D method generates — NOT a separate scalar-field
    proxy. A genuine fold shows as a self-touching / closed curve (det J ≤ 0),
    which is precisely what we want a debug section to reveal.

    `Φ` is defined on the whole grid (deform3d extends it into the air as the
    nearest in-model displacement), so the raw contours run out into the air;
    we clip them to the model to keep only real in-part layers. Returns
    (surface, z_lo, z_hi) — the deformed-Z range over the model, for the clim."""
    nx, ny, nz = gr.step.shape
    phiz = np.ascontiguousarray(dmap.phi[..., 2], dtype=np.float32)
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(dmap.pitch, dmap.pitch, dmap.pitch),
        origin=(float(dmap.origin[0]), float(dmap.origin[1]), float(dmap.origin[2])),
    )
    grid.cell_data["z_def"] = phiz.flatten(order="F")
    pgrid = grid.cell_data_to_point_data()

    inside = gr.step >= 0
    zin = phiz[inside]
    lo, hi = float(zin.min()), float(zin.max())
    step = layer_step if (layer_step and layer_step > 0) else dmap.pitch
    n = max(1, int(round((hi - lo) / step)))
    isos = [lo + (i + 0.5) * step for i in range(n)]
    isos = [c for c in isos if lo < c < hi] or [0.5 * (lo + hi)]
    surf = pgrid.contour(isosurfaces=isos, scalars="z_def")

    # Drop deform3d's air extrapolation: keep only the layers inside the part.
    if surf.n_points and mesh is not None:
        try:
            surf = surf.clip_surface(pv.wrap(mesh), invert=True)
        except Exception:  # noqa: BLE001 — implicit clip can fail on odd meshes
            pass
    return surf, lo, hi


def save_xz_section(
    mesh: trimesh.Trimesh,
    out_path: str,
    *,
    pitch: float = 1.0,
    max_tilt_deg: float = 30.0,
    smooth_sigma: float = 2.0,
    depth_method: str = "vectors",
    deform_mode: str = "z-only",
    section_y: float | None = None,
    window_size: tuple[int, int] = (1600, 1000),
    line_width: float = 2.5,
    cmap: str = "turbo",
) -> int:
    """Compute growth, build the layer surfaces, cut them with the XZ plane,
    and save an orthographic image of the resulting curves to `out_path`.

    `deform_mode` picks WHAT the section shows, matching the deformation path:
      - "3d": the ACTUAL layers of the full-3D map — iso-surfaces of the
        deformed-Z field Φ_z (build_deform_layer_surfaces). A fold shows as a
        closed/self-touching curve. This is the faithful debug view.
      - "z-only" (default): iso-surfaces of the scalar depth field
        (build_growth_surfaces) — what the Z-only path uses.

    Returns the number of points in the section (0 means the plane missed
    the geometry — bad `section_y`). The slice is taken at the model's Y
    centre unless `section_y` is given.
    """
    vg = voxelize_solid(mesh, pitch=pitch)
    gr = compute_growth(vg, max_tilt_deg=max_tilt_deg)
    if deform_mode == "3d":
        from .deform3d import solve_deformation_map
        dmap = solve_deformation_map(
            gr, displacement_smooth_sigma=smooth_sigma, max_tilt_deg=max_tilt_deg,
            depth_method=depth_method,
        )
        surf, z_lo, z_hi = build_deform_layer_surfaces(dmap, gr, mesh)
        scalar_name, clim = "z_def", [z_lo, z_hi]
    else:
        surf = build_growth_surfaces(gr, smooth_sigma=smooth_sigma, depth_method=depth_method)
        scalar_name, clim = "step", [0.0, max(float(gr.n_steps - 1), 1.0)]

    cx = 0.5 * (mesh.bounds[0, 0] + mesh.bounds[1, 0])
    cz = 0.5 * (mesh.bounds[0, 2] + mesh.bounds[1, 2])
    cy = section_y if section_y is not None else 0.5 * (mesh.bounds[0, 1] + mesh.bounds[1, 1])
    diag = float(np.linalg.norm(mesh.extents)) or 1.0

    section = surf.slice(normal=(0.0, 1.0, 0.0), origin=(cx, cy, cz))

    # The model's own outline at the SAME plane: slicing the closed surface mesh
    # with the XZ plane gives its boundary polylines (the part's cross-section
    # contour). Drawn over everything below in a contrasting colour.
    model_outline = pv.wrap(mesh).slice(normal=(0.0, 1.0, 0.0), origin=(cx, cy, cz))

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background("white")
    if section.n_points > 0:
        pl.add_mesh(
            section,
            scalars=scalar_name,
            cmap=cmap,
            line_width=line_width,
            show_scalar_bar=False,
            lighting=False,
            clim=clim,
        )
    # Model contour ON TOP of the layer curves. The view is orthographic down
    # +Y with the camera on the +Y side, so nudging the outline toward the
    # camera (+Y) makes it win the depth test against the coplanar layer lines
    # without moving it one pixel on screen (Y is the view axis → no XZ shift).
    if model_outline.n_points > 0:
        model_outline.points[:, 1] += 0.01 * diag
        pl.add_mesh(
            model_outline,
            color="black",
            line_width=line_width + 1.0,
            show_scalar_bar=False,
            lighting=False,
        )
    # Orthographic, looking straight down the +Y axis: X horizontal, Z up.
    # Set the camera explicitly — view_xz() picks an axis mapping that
    # rotates the section 90°.
    pl.enable_parallel_projection()
    pl.camera_position = [
        (cx, cy + 2.0 * diag, cz),   # camera on the +Y side
        (cx, cy, cz),                # looking at the section
        (0.0, 0.0, 1.0),             # Z is up
    ]
    pl.reset_camera()
    pl.screenshot(out_path)
    pl.close()
    return int(section.n_points)
