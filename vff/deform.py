"""Mesh deformation: remap a 3D mesh so the growth iso-surfaces become
parallel XY planes.

Pipeline target: feed the deformed mesh to a vanilla planar slicer. After
the planar slicer cuts horizontal layers in the deformed space, those
cuts correspond one-to-one with the curved non-planar growth surfaces in
the original space.

How:
  1. For each mesh vertex v at world position (x, y, z), trilinearly
     sample the integer-valued `step` field at v. Outside the model the
     step is -1; we extend it via a nearest-painted-voxel lookup so every
     vertex gets a sane step value.
  2. new_z = step_continuous * dz_per_layer. X and Y are preserved.

This is the simplest faithful mapping: it makes every growth iso-surface
land on a single horizontal plane (z = N * dz_per_layer for step N). It
does NOT try to preserve in-plane distances — that would require solving
an integration / Poisson problem on the surface; we'll revisit if a
distortion-aware deformation is needed downstream.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

from .growth import GrowthResult


def smoothed_depth_field(growth: GrowthResult, sigma: float = 1.5) -> np.ndarray:
    """Continuous "depth from bed" field used by both surface viz and deform.

    Combines two ideas:

      1. Outside the model the field is VERTICAL: `field = k - k_bed_layer`.
         So iso-surfaces in empty space are exact horizontal planes — the
         normals there point straight up, matching the print head's default
         orientation. No more radial flood-fill perturbations near the model.

      2. Inside the model the field starts as the integer BFS step, which is
         anisotropic — it depends on how the model's features happen to line
         up with the voxel grid axes. Two geometrically-identical features
         at different angles (e.g. propeller blades) end up with slightly
         different step distributions, which makes their deformations look
         different. A Gaussian smoothing of the whole field (interior +
         vertical exterior, jointly) averages this anisotropy out and gives
         a continuous "depth" that respects the model's symmetry.

    Key boundary property: inside the model at the bed (k = k_bed_layer)
    step is 0; the vertical extension at the same k is also 0 (=
    k - k_bed_layer). They agree on the bed surface, so Gaussian smoothing
    doesn't introduce a gradient across the bed boundary — bed contact is
    preserved.

    `sigma` is in voxel units. Default 1.5 = light smoothing that visibly
    symmetrizes the field without erasing real growth geometry. Heavier
    sigma symmetrizes more but starts to round off real overhangs.
    """
    step = growth.step
    if step.size == 0 or not (step >= 0).any():
        return np.zeros(step.shape, dtype=np.float32)

    nx, ny, nz = step.shape

    # k_bed_layer = lowest Z-layer that contains any model voxel.
    # Matches _seed_bed_mask convention (step=0 voxels live here).
    has_model_in_z = (step >= 0).any(axis=(0, 1))
    k_bed_layer = int(np.argmax(has_model_in_z))

    # Initialize with BFS step inside the model.
    field = step.astype(np.float32).copy()

    # Vertical extension outside the model: step = k - k_bed_layer.
    k_axis = np.arange(nz, dtype=np.float32) - float(k_bed_layer)
    k_grid = np.broadcast_to(k_axis[None, None, :], step.shape)
    empty = step < 0
    field[empty] = k_grid[empty]

    # Joint Gaussian smoothing across interior + vertical exterior.
    if sigma > 0:
        field = gaussian_filter(field, sigma=sigma, mode="nearest")

    return field


def _extended_step_field(growth: GrowthResult) -> np.ndarray:
    """Return a copy of growth.step where -1 cells have been replaced by the
    step of the nearest painted voxel. Done once per growth, then reused
    for cheap trilinear sampling.

    Implementation: iterative 26-conn dilation that propagates step values
    outward from painted voxels. Stops when the whole array is filled.
    Linear in the number of empty cells; for our typical grid sizes
    (135 x 135 x 26) it runs in well under a second."""
    step = growth.step.copy()
    if (step < 0).sum() == 0:
        return step.astype(np.float32)

    # Pad with one ring on each side so 26-neighbour lookups stay in-bounds.
    nx, ny, nz = step.shape
    padded = np.full((nx + 2, ny + 2, nz + 2), -1, dtype=np.int32)
    padded[1:-1, 1:-1, 1:-1] = step

    # Iterate: each round each empty cell takes a value from any non-empty
    # 26-neighbour. Stop when nothing changes.
    while True:
        empty = padded < 0
        if not empty.any():
            break
        # For each of the 26 offsets, where the offset points at a non-empty
        # cell, copy that value into the empty cell.
        changed = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    src = padded[1 + dx: 1 + dx + nx, 1 + dy: 1 + dy + ny, 1 + dz: 1 + dz + nz]
                    dst = padded[1: 1 + nx, 1: 1 + ny, 1: 1 + nz]
                    mask = (dst < 0) & (src >= 0)
                    if mask.any():
                        dst[mask] = src[mask]
                        changed = True
        if not changed:
            # Should not happen unless the grid had no painted voxels at all,
            # which we handled above. Bail to avoid infinite loop.
            break

    return padded[1:-1, 1:-1, 1:-1].astype(np.float32)


def _sample_trilinear(field: np.ndarray, origin: np.ndarray, pitch: float, points: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of `field` at world `points`. Cells of the
    field are pitch-sized and the cell-(0,0,0) corner sits at `origin`.
    Out-of-bounds points clamp to the boundary cell value."""
    nx, ny, nz = field.shape

    # World -> cell-center coordinates.
    p = (points - origin) / pitch - 0.5  # cell (i,j,k) center -> integer coordinate i

    # Clamp into the inner grid so the i+1 neighbour exists. Beyond the grid
    # we extrapolate to the boundary value.
    p[:, 0] = np.clip(p[:, 0], 0.0, nx - 1.0001)
    p[:, 1] = np.clip(p[:, 1], 0.0, ny - 1.0001)
    p[:, 2] = np.clip(p[:, 2], 0.0, nz - 1.0001)

    i0 = np.floor(p[:, 0]).astype(np.int64)
    j0 = np.floor(p[:, 1]).astype(np.int64)
    k0 = np.floor(p[:, 2]).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    fx = (p[:, 0] - i0).astype(np.float32)
    fy = (p[:, 1] - j0).astype(np.float32)
    fz = (p[:, 2] - k0).astype(np.float32)

    c000 = field[i0, j0, k0]
    c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]
    c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]
    c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]
    c111 = field[i1, j1, k1]

    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fz) + c1 * fz


def deform_mesh(
    mesh: trimesh.Trimesh,
    growth: GrowthResult,
    dz_per_layer: float | None = None,
    bed_z: float = 0.0,
    bed_blend_height: float | None = None,
    smooth_sigma: float = 1.5,
) -> trimesh.Trimesh:
    """Return a deformed copy of `mesh` whose Z is driven by the growth step.

    Math:
        step_c = trilinear sample of growth.step (extended) at vertex
        z_target = step_c * dz_per_layer + bed_z

        w = clamp((orig_z - bed_z) / bed_blend_height, 0, 1)
        new_z = (1 - w) * orig_z + w * z_target

    Why the blend: pure step-based mapping `new_z = step_c * dz` lifts
    bed-touching vertices off the bed when their neighbours-in-XY are
    outside the model (the extended step field there is non-zero, dragging
    the trilinear sample up). Empirically observed: at pitch=0.5mm on the
    propeller, bed-vertices were spreading over 0.75mm in deformed Z.

    The smooth blend pins `new_z = orig_z` exactly at the bed (orig_z =
    bed_z) and ramps to full step-based mapping over `bed_blend_height`.
    Default blend height = 2 * pitch (two voxel layers), enough to absorb
    sampling artifacts without distorting the macro shape.

    XY is left untouched in this pass; in-plane distortion is a follow-up
    if the downstream slicer needs it (see RotBotSlicer's refinement step
    or S4_Slicer's per-tet optimization for principled approaches)."""
    if dz_per_layer is None:
        dz_per_layer = growth.pitch
    if bed_blend_height is None:
        bed_blend_height = 2.0 * growth.pitch

    # smoothed_depth_field gives a continuous, symmetry-respecting field that
    # agrees with the BFS step inside the model and with vertical depth
    # outside it. See its docstring for why this matters for symmetric parts.
    field = smoothed_depth_field(growth, sigma=smooth_sigma)
    verts = mesh.vertices.astype(np.float64, copy=False)
    step_continuous = _sample_trilinear(field, growth.origin, growth.pitch, verts).astype(np.float64)

    z_target = step_continuous * dz_per_layer + bed_z
    z_orig = verts[:, 2]

    if bed_blend_height > 0:
        w = np.clip((z_orig - bed_z) / bed_blend_height, 0.0, 1.0)
    else:
        w = np.ones_like(z_orig)
    new_z = (1.0 - w) * z_orig + w * z_target

    new_verts = verts.copy()
    new_verts[:, 2] = new_z

    return trimesh.Trimesh(
        vertices=new_verts.astype(np.float64),
        faces=mesh.faces,
        process=False,
    )
