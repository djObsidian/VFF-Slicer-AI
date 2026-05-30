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
import numpy.ma as ma
import trimesh
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import coo_array, csr_array
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import cg

try:
    import skfmm  # type: ignore
    _HAVE_SKFMM = True
except ImportError:
    _HAVE_SKFMM = False

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


def fmm_distance_from_bed(growth: GrowthResult) -> np.ndarray:
    """Fast Marching Method (Eikonal solver) for geodesic distance from the
    bed-seed surface, restricted to model interior.

    Returns float32 (nx,ny,nz). Cells outside the model and unreachable
    cells get np.inf.

    FMM solves ||∇d|| = 1 with d=0 at the front (bed seeds). The result is
    the continuous-equivalent of weighted-Dijkstra geodesic distance, but
    typically smoother near the wavefront-merge surfaces because FMM
    propagates a continuous wave instead of a discrete shortest-path
    relaxation. Requires scikit-fmm.
    """
    if not _HAVE_SKFMM:
        raise RuntimeError("scikit-fmm not installed; pip install scikit-fmm")
    step = growth.step
    matrix = step >= 0
    if not matrix.any():
        return np.full(matrix.shape, np.inf, dtype=np.float32)

    has_in_z = matrix.any(axis=(0, 1))
    k_bed = int(np.argmax(has_in_z))
    bed_mask = np.zeros_like(matrix)
    bed_mask[:, :, k_bed] = matrix[:, :, k_bed]

    # phi: negative at bed-seed cells (inside the front), positive elsewhere.
    # Mask non-model cells so the wave can't propagate through air.
    phi = np.full(matrix.shape, 1.0, dtype=np.float64)
    phi[bed_mask] = -1.0
    phi_ma = ma.MaskedArray(phi, mask=~matrix)

    dist = skfmm.distance(phi_ma, dx=growth.pitch)
    if isinstance(dist, ma.MaskedArray):
        dist = dist.filled(np.inf)
    return dist.astype(np.float32)


def integrate_vectors_to_potential(growth: GrowthResult, eps: float = 1e-6) -> np.ndarray:
    """Least-squares scalar potential φ whose gradient best matches the
    (clamped) growth vector field. The level sets of φ then have the growth
    vectors as their normals — i.e. each layer surface is perpendicular to
    the local growth direction, as closely as a single globally-consistent
    surface family allows.

    This is the "integrate the vectors into a surface" route: instead of
    deriving the depth from geodesic distance (FMM/Dijkstra) and clamping
    only the *display* arrows, we build the depth field FROM the clamped
    vectors. The nozzle-tilt clamp therefore actually shapes the layers.

    Method (discrete Poisson / "surface from gradients"):

      For every 26-connectivity edge (i, j) between model voxels we want the
      potential difference to equal the growth vector projected on the edge:

          φ_j − φ_i  ≈  v̄_ij · (x_j − x_i)        v̄_ij = ½(v_i + v_j)

      Least-squares over all edges gives the normal equations  L φ = b  with
      L the (weighted) graph Laplacian and b the discrete divergence of the
      target field. Bed-seed voxels are pinned to φ = 0 (Dirichlet, via a
      large diagonal penalty) so numbering starts at the plate and grows
      upward; a tiny Tikhonov term `eps` keeps disconnected components
      solvable. Solved with conjugate gradient (SPD system).

    Exactness caveat: a clamped field is generally NOT a gradient field, so
    no surface has these vectors as *exact* normals everywhere. φ is the
    closest consistent compromise — the residual ‖∇φ − v‖ is where the tilt
    clamp fought global consistency.

    Returns: float32 grid (nx, ny, nz), values in voxel-step units inside
    the model, +inf outside (filled in later by smoothed_depth_field).
    """
    matrix = growth.step >= 0
    nx, ny, nz = matrix.shape
    n_model = int(matrix.sum())
    if n_model == 0:
        return np.full(matrix.shape, np.inf, dtype=np.float32)

    # Compact model-voxel indexing 0..M-1 in C order (matches boolean masking).
    order = np.argwhere(matrix).astype(np.int64)          # (M, 3) voxel coords
    flat_idx = np.full(matrix.shape, -1, dtype=np.int64)
    flat_idx[matrix] = np.arange(n_model)
    vecs = growth.vectors[matrix].astype(np.float64)      # (M, 3) clamped growth dirs

    # 13 canonical offsets — one from each ±pair, so every edge is built once.
    canon = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx, dy, dz) > (0, 0, 0)
    ]

    rows_l, cols_l, vals_l = [], [], []
    b = np.zeros(n_model, dtype=np.float64)
    for off in canon:
        d = np.asarray(off, dtype=np.float64)
        nbr = order + np.asarray(off, dtype=np.int64)
        inb = (
            (nbr[:, 0] >= 0) & (nbr[:, 0] < nx)
            & (nbr[:, 1] >= 0) & (nbr[:, 1] < ny)
            & (nbr[:, 2] >= 0) & (nbr[:, 2] < nz)
        )
        src = np.where(inb)[0]
        if src.size == 0:
            continue
        nb = nbr[src]
        jj = flat_idx[nb[:, 0], nb[:, 1], nb[:, 2]]
        valid = jj >= 0
        if not valid.any():
            continue
        i_arr = src[valid]                                # model index of voxel i
        j_arr = jj[valid]                                 # model index of voxel j
        vbar = 0.5 * (vecs[i_arr] + vecs[j_arr])
        t = vbar @ d                                      # target φ_j − φ_i
        ones = np.ones(i_arr.size, dtype=np.float64)
        # Laplacian contribution of edge (i,j): +1 on diagonals, −1 off.
        rows_l += [i_arr, j_arr, i_arr, j_arr]
        cols_l += [i_arr, j_arr, j_arr, i_arr]
        vals_l += [ones, ones, -ones, -ones]
        # Divergence RHS: equation (φ_j − φ_i = t) pushes b[i]-=t, b[j]+=t.
        np.add.at(b, i_arr, -t)
        np.add.at(b, j_arr, t)

    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    lap = coo_array((vals, (rows, cols)), shape=(n_model, n_model)).tocsr()

    # Pin bed seeds to φ = 0 (lowest non-empty Z layer — same convention as
    # growth/geodesic). Large diagonal penalty ≈ Dirichlet; eps elsewhere
    # regularizes the (otherwise singular, constant-nullspace) Laplacian.
    has_in_z = matrix.any(axis=(0, 1))
    k_bed = int(np.argmax(has_in_z))
    seed_grid = np.zeros_like(matrix)
    seed_grid[:, :, k_bed] = matrix[:, :, k_bed]
    seed_local = flat_idx[seed_grid]
    diag_mean = float(lap.diagonal().mean())
    big = 1.0e6 * (diag_mean if diag_mean > 0 else 1.0)
    pen = np.full(n_model, eps, dtype=np.float64)
    pen[seed_local] = big
    rng = np.arange(n_model)
    a_mat = (lap + coo_array((pen, (rng, rng)), shape=(n_model, n_model))).tocsr()
    # Seed target is 0, so b stays 0 there; the penalty drags φ_seed → 0.

    try:
        phi, info = cg(a_mat, b, rtol=1e-7, maxiter=5000)
    except TypeError:  # SciPy < 1.12 used `tol` instead of `rtol`.
        phi, info = cg(a_mat, b, tol=1e-7, maxiter=5000)
    if info != 0:
        # CG didn't converge (rare at our sizes) — fall back to a direct solve.
        from scipy.sparse.linalg import spsolve
        phi = spsolve(a_mat, b)

    out = np.full(matrix.shape, np.inf, dtype=np.float32)
    out[matrix] = phi.astype(np.float32)
    return out


def smoothed_depth_field(
    growth: GrowthResult,
    sigma: float = 2.0,
    method: str = "fmm",
    outside_mode: str = "extend",
) -> np.ndarray:
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

    # Inside-model depth.
    #   "vectors"  — least-squares scalar potential whose gradient matches the
    #                CLAMPED growth vectors (integrate_vectors_to_potential).
    #                Layer surfaces (its level sets) are perpendicular to the
    #                growth direction, so the nozzle-tilt clamp actually shapes
    #                them. This is the DEFAULT.
    #   "fmm"      — Eikonal solver via scikit-fmm. Smooth (C1) where it
    #                exists, including across wavefront-merge surfaces.
    #                Falls back to "dijkstra" if scikit-fmm isn't installed.
    #                Ignores the clamp — surfaces follow raw geodesic depth.
    #   "dijkstra" — discrete shortest path on the 26-conn voxel graph.
    #                C0 only, can show small kinks at cell boundaries.
    if method == "vectors":
        geo = integrate_vectors_to_potential(growth)
    elif method == "fmm" and _HAVE_SKFMM:
        geo = fmm_distance_from_bed(growth)
    else:
        geo = geodesic_distance_from_bed(growth)
    field = geo.astype(np.float32, copy=True)

    outside = step < 0
    k_axis = np.arange(nz, dtype=np.float32) - float(k_bed_layer)
    k_grid = np.broadcast_to(k_axis[None, None, :], step.shape)

    # Outside-model handling — picked to match the consumer:
    #
    #   "extend"   — each outside cell inherits the depth of its nearest
    #                MODEL cell (Euclidean nearest via distance_transform_edt).
    #                The field is CONTINUOUS across the model boundary, so a
    #                mesh vertex sitting on the model surface gets the same
    #                depth from inside cells and from adjacent air cells
    #                via trilinear. Default — required by the deformation
    #                path; without it, blade-tip vertices bridge a sudden
    #                inside↔outside depth gap and the mesh spikes outward.
    #
    #   "vertical" — outside cells get k - k_bed_layer (linear in z). Iso-
    #                surfaces in air become exact horizontal planes with
    #                vertical normals. Use this for SURFACE VISUALISATION,
    #                where the user wants the air part of each iso-surface
    #                to look flat. NOT for deformation.
    if outside.any():
        if outside_mode == "extend":
            idx = distance_transform_edt(
                outside, return_distances=False, return_indices=True
            )
            field[outside] = field[idx[0], idx[1], idx[2]][outside]
        elif outside_mode == "vertical":
            field[outside] = k_grid[outside]
        else:
            raise ValueError(
                f"outside_mode must be 'extend' or 'vertical', got {outside_mode!r}"
            )

    # Disconnected components — bed Dijkstra/FMM never reaches them, so
    # they stay at inf. Backfill with the vertical default so they at
    # least get a sane value (rare but happens with split models).
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
    smooth_sigma: float = 2.0,
    depth_method: str = "fmm",
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
    field = smoothed_depth_field(growth, sigma=smooth_sigma, method=depth_method)
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
