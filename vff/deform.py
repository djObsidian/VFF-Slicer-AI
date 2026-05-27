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
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra

from .growth import GrowthResult


_NBR_OFFSETS_WEIGHTED = np.array(
    [(dx, dy, dz, float(np.sqrt(dx * dx + dy * dy + dz * dz)))
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if (dx, dy, dz) != (0, 0, 0)],
    dtype=np.float64,
)


def geodesic_distance_from_bed(growth: GrowthResult) -> np.ndarray:
    """Multi-source weighted Dijkstra shortest-path distance from bed seeds
    through model voxels using 26-connectivity, edge weight = Euclidean
    voxel distance (1, √2, √3 for face/edge/vertex neighbours).

    Returns a float32 grid of geodesic distances (in voxel units) for
    model voxels, np.inf for everything outside the model. Compared to
    the integer BFS step:
      - continuous (no quantization)
      - reflects actual path length (corner-cutting via diagonals is
        properly more expensive than going around the face)
      - typically gives larger max values than BFS step for parts with
        diagonal-dominated chains (stronger deformation downstream)
    """
    # Model voxels = those with assigned step (>= 0). Outside is -1.
    matrix = growth.step >= 0
    nx, ny, nz = matrix.shape

    # Compact model-voxel indexing: model voxels get 0..M-1, others -1.
    n_voxels = int(matrix.sum())
    if n_voxels == 0:
        return np.full(matrix.shape, np.inf, dtype=np.float32)
    flat_idx = np.full(matrix.shape, -1, dtype=np.int64)
    flat_idx[matrix] = np.arange(n_voxels)

    model_indices = np.argwhere(matrix).astype(np.int64)  # (M, 3)

    # Build adjacency in COO form.
    rows_all = []
    cols_all = []
    wts_all = []
    for dx, dy, dz, w in _NBR_OFFSETS_WEIGHTED:
        d = np.array([dx, dy, dz], dtype=np.int64)
        nbr = model_indices + d
        inb = (
            (nbr[:, 0] >= 0) & (nbr[:, 0] < nx)
            & (nbr[:, 1] >= 0) & (nbr[:, 1] < ny)
            & (nbr[:, 2] >= 0) & (nbr[:, 2] < nz)
        )
        if not inb.any():
            continue
        in_pos = np.where(inb)[0]
        nbr_in = nbr[in_pos]
        nbr_flat = flat_idx[nbr_in[:, 0], nbr_in[:, 1], nbr_in[:, 2]]
        valid = nbr_flat >= 0
        if not valid.any():
            continue
        from_idx = in_pos[valid]
        to_idx = nbr_flat[valid]
        rows_all.append(from_idx)
        cols_all.append(to_idx)
        wts_all.append(np.full(len(from_idx), w, dtype=np.float64))

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    wts = np.concatenate(wts_all)
    graph = csr_array((wts, (rows, cols)), shape=(n_voxels, n_voxels))

    # Seeds: lowest non-empty Z layer's filled voxels (same convention as growth).
    has_in_z = matrix.any(axis=(0, 1))
    if not has_in_z.any():
        return np.full(matrix.shape, np.inf, dtype=np.float32)
    k_seed = int(np.argmax(has_in_z))
    seed_mask = np.zeros(matrix.shape, dtype=bool)
    seed_mask[:, :, k_seed] = matrix[:, :, k_seed]
    seed_flat = flat_idx[seed_mask]

    # Multi-source Dijkstra: scipy's min_only=True does this in a SINGLE
    # Dijkstra run (treats all seeds as zero-cost starting points), instead
    # of running one Dijkstra per source. Two orders of magnitude faster
    # when there are many seeds (propeller has ~1.8k bed seeds @ pitch=1).
    dist_flat = dijkstra(
        graph,
        directed=False,
        indices=seed_flat,
        return_predecessors=False,
        min_only=True,
    )

    dist = np.full(matrix.shape, np.inf, dtype=np.float32)
    dist[matrix] = dist_flat.astype(np.float32)
    return dist


def smoothed_depth_field(growth: GrowthResult, sigma: float = 1.0) -> np.ndarray:
    """Continuous "depth from bed" field used by both surface viz and deform.

    Construction:

      1. Inside the model: **weighted-Dijkstra geodesic distance from bed**
         (face/edge/vertex edges with weights 1/√2/√3). Continuous,
         direction-symmetric, and naturally larger than the integer BFS
         step for chains that go through diagonals — so the downstream
         deformation actually stretches.

      2. Outside the model: vertical depth `field = k - k_bed_layer`.
         Iso-surfaces in empty space are exact horizontal planes →
         normals point straight up. No flood-fill perturbations.

      3. Light Gaussian smoothing (default sigma=0.6 voxels) to take the
         edge off the residual BFS-axis flavour in the geodesic field.
         Less aggressive than before — the geodesic field is already much
         smoother than the integer step it replaces.

    Boundary continuity: at the bed (k=k_bed_layer) the geodesic distance
    is 0 (seed voxels) and the vertical extension is 0 (k - k_bed_layer).
    They agree, so smoothing doesn't drag bed-touching values up.
    """
    step = growth.step
    if step.size == 0 or not (step >= 0).any():
        return np.zeros(step.shape, dtype=np.float32)

    nx, ny, nz = step.shape

    has_model_in_z = (step >= 0).any(axis=(0, 1))
    k_bed_layer = int(np.argmax(has_model_in_z))

    # Inside-model: weighted geodesic distance from bed seeds.
    geo = geodesic_distance_from_bed(growth)  # float32, inf outside model
    field = geo.astype(np.float32, copy=True)

    # Outside-model: vertical depth k - k_bed_layer.
    k_axis = np.arange(nz, dtype=np.float32) - float(k_bed_layer)
    k_grid = np.broadcast_to(k_axis[None, None, :], step.shape)
    outside = step < 0
    field[outside] = k_grid[outside]

    # Replace any residual inf (e.g. disconnected model components that bed
    # never reaches) with the vertical extension value at that cell.
    bad = ~np.isfinite(field)
    if bad.any():
        field[bad] = k_grid[bad]

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
    smooth_sigma: float = 1.0,
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
