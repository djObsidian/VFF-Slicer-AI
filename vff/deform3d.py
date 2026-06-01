"""Full-3D "straightening" deformation — the principled successor to the
Z-only deform_mesh.

WHY (vs deform.py's Z-only map):
    The clamped growth vector field g(p) is the local build direction. We want
    a deformation Φ: original → deformed whose local *rotation* takes g → +ẑ
    everywhere, so the growth iso-surfaces become horizontal planes a planar
    slicer can cut. Z-only deform moves only Z, which shears in-plane distances
    on tilted layers and can't represent a surface that folds over XY. The 3D
    map moves all three axes (ARAP-style), preserving local distances.

FORMULATION (discrete Poisson "deformation from a rotation field" — the
3-coordinate generalization of deform.integrate_vectors_to_potential, reusing
the same graph Laplacian):

    - Per model voxel i:  R_i = minimal rotation taking ĝ_i → ẑ
      (identity where g_i ≈ ẑ or g_i = 0, e.g. bed seeds / outside).
    - Per 26-connectivity edge (i, j) between model voxels:
          target offset  t_ij = R̄_ij · (p_j − p_i),   R̄_ij = ½(R_i + R_j)
      and we want   Φ_j − Φ_i ≈ t_ij.
    - Least squares over all edges, independently per output coordinate c:
          L Φ^c = b^c,   L = weighted graph Laplacian (shared across c),
          b^c = discrete divergence of t^c.
    - Bed-seed voxels are pinned (Dirichlet, via a large diagonal penalty) to
      their ORIGINAL world positions, so the plate footprint stays put (bed
      adhesion) and the per-coordinate translation gauge is fixed.

    Exactness caveat (same as the scalar version): a clamped field is generally
    not a gradient field, so no Φ has these rotations as exact local frames
    everywhere. Φ is the closest globally-consistent compromise; the residual
    is where the tilt clamp fought global integrability.

MAPS:
    forward_map  — trilinear sample of Φ at an original-space point (the
                   original grid is regular, so this is the easy direction).
    inverse_map  — vectorised Newton on the trilinear Φ field (find p with
                   Φ(p) = q), using precomputed Jacobian fields ∂Φ/∂{x,y,z}.
                   Converges in a handful of iterations on a fold-free Φ;
                   where Φ folds (overhangs / self-intersection — the known
                   limitation) Newton fails to converge and we flag/clamp it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_array
from scipy.sparse.linalg import cg, spsolve
from scipy.spatial.transform import Rotation

from .growth import GrowthResult


# 13 canonical 26-conn offsets — one per ± pair, so each edge is built once.
_CANON_OFFSETS = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if (dx, dy, dz) > (0, 0, 0)],
    dtype=np.int64,
)


def rotations_to_vertical(vectors: np.ndarray) -> np.ndarray:
    """Per-vector minimal rotation matrix taking each (clamped growth) vector
    to +ẑ. `vectors` is (M, 3); returns (M, 3, 3).

    For a unit ĝ the rotation is about axis ĝ × ẑ ∝ (g_y, −g_x, 0) by angle
    arccos(g_z). Zero/near-vertical vectors map to identity (rotvec → 0)."""
    v = np.asarray(vectors, dtype=np.float64)
    mag = np.linalg.norm(v, axis=1)
    out = np.zeros((v.shape[0], 3), dtype=np.float64)  # rotvecs
    active = mag > 1e-9
    if active.any():
        u = v[active] / mag[active, None]
        gz = np.clip(u[:, 2], -1.0, 1.0)
        angle = np.arccos(gz)                      # (m,)
        axis = np.stack([u[:, 1], -u[:, 0], np.zeros_like(gz)], axis=1)
        ax_norm = np.linalg.norm(axis, axis=1)
        good = ax_norm > 1e-9                       # not already ±ẑ
        rv = np.zeros_like(u)
        rv[good] = (axis[good] / ax_norm[good, None]) * angle[good, None]
        # Vectors already ~+ẑ → angle≈0 → identity (rv stays 0). Vectors ~−ẑ
        # (shouldn't occur after a <90° tilt clamp) fall back to identity too.
        out[active] = rv
    return Rotation.from_rotvec(out).as_matrix().astype(np.float64)


def _nearest_good_fill(values: np.ndarray, good: np.ndarray) -> np.ndarray:
    """For each row, return `values` of the nearest row flagged True in `good`,
    measured in index order (ties → the earlier one). Rows with no good row
    anywhere are returned unchanged. Used to warm-start the inverse's leftover
    non-converged points from a converged neighbour along the toolpath."""
    N = values.shape[0]
    gi = np.where(good)[0]
    if gi.size == 0:
        return values.copy()
    idxs = np.arange(N)
    pos = np.searchsorted(gi, idxs)
    left = gi[np.clip(pos - 1, 0, gi.size - 1)]
    right = gi[np.clip(pos, 0, gi.size - 1)]
    nearest = np.where(np.abs(idxs - left) <= np.abs(idxs - right), left, right)
    return values[nearest]


@dataclass
class DeformationMap:
    """Φ sampled on the (regular) original voxel grid plus its Jacobian.

    phi[i,j,k] = deformed world position of original grid node (i,j,k).
    The original node world position is origin + (idx + 0.5) * pitch.
    """
    phi: np.ndarray          # (nx, ny, nz, 3) float64 — deformed positions
    jac: np.ndarray          # (nx, ny, nz, 3, 3) float64 — d(phi)/d(world xyz)
    origin: np.ndarray       # (3,) world corner of node (0,0,0)
    pitch: float

    # ---- forward: original → deformed (trilinear) ----
    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        return _sample_vec(self.phi, self.origin, self.pitch, np.asarray(pts, float))

    # ---- local volume scaling det(JΦ) at original-space points ----
    def jacobian_det(self, pts: np.ndarray) -> np.ndarray:
        """det of the forward Jacobian ∂Φ/∂(world) at each point = dV_def/dV_orig
        (the local volume stretch of the deformation). VOLUME extrusion comp
        (4/5-axis, S4-style): the slicer's E assumes the deformed road volume, so
        the original-space deposit scales by 1/det."""
        J = _sample_jac(self.jac, self.origin, self.pitch, np.asarray(pts, float))
        return np.linalg.det(J)

    def layer_gap_ratio(self, pts: np.ndarray) -> np.ndarray:
        """∂(original z)/∂(deformed z) = (JΦ⁻¹)[z,z] at each point — how a
        deformed-space vertical layer step maps to original-space vertical
        spacing. VERTICAL extrusion comp (3-axis, vertical nozzle, fixed road
        WIDTH): over/under-extrusion is driven by the layer-height squish, not
        the full volume. The slicer's E assumes the deformed layer height; the
        real gap to the layer below is this × that, so E scales by it (ratio < 1
        where layers compress → reduce E to avoid over-squish). Unlike 1/det
        this ignores the in-plane width change a 3-axis nozzle can't realize
        (valid because the 3D map is near-isometric in-plane)."""
        J = _sample_jac(self.jac, self.origin, self.pitch, np.asarray(pts, float))
        # (J⁻¹)[z,z] = cofactor_zz / det = (Jxx·Jyy − Jxy·Jyx) / det. Computed
        # directly rather than via np.linalg.inv: a full batch-inverse RAISES
        # LinAlgError if ANY sampled J is singular (a non-converged inverse point
        # at a fold can hit one), which would kill the whole gcode transform.
        # This is identical for non-singular J, ~3× cheaper, and falls back to
        # 1.0 (no compensation) on a degenerate cell instead of crashing.
        det = np.linalg.det(J)
        cof_zz = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        out = np.ones(J.shape[0], dtype=np.float64)
        ok = np.abs(det) > 1e-9
        out[ok] = cof_zz[ok] / det[ok]
        return out

    # ---- inverse: deformed → original (vectorised damped Newton / LM) ----
    def inverse_points(
        self, q: np.ndarray, iters: int = 30, tol: float = 1e-4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (p, converged_mask). p[k] satisfies Φ(p[k]) ≈ q[k].

        Solved as a per-point least-squares ‖Φ(p) − q‖² with an **adaptive
        Levenberg–Marquardt** step (a damped Gauss–Newton):

            (JᵀJ + λ·diag(JᵀJ)) Δp = Jᵀ(Φ(p) − q),   p ← clip(p − Δp).

        Each point carries its own λ with a built-in line search: a step that
        lowers the residual is accepted and λ shrinks (→ undamped Newton, fast
        quadratic convergence on the fold-free bulk); a step that doesn't is
        rejected and λ grows (→ a short, safe gradient step). This is what fixes
        the tips: plain Newton (the old code) FROZE near-singular Jacobians at
        folds — those points never moved and showed up as the ~2 % that didn't
        converge; LM instead damps the singular direction and keeps making
        progress, and where Φ genuinely folds (non-injective) the step can't
        reduce the residual so the point simply stays at its best iterate.

        We track the BEST iterate per point (not the last): a truly stuck point
        otherwise drifts to the grid-clamp boundary and returns Z = grid-top
        garbage (the old wedge-spikes-to-the-ceiling). `converged_mask` is False
        only where even the best iterate didn't reach `tol`."""
        q = np.asarray(q, dtype=np.float64)
        N = q.shape[0]
        if N == 0:
            return q.copy(), np.zeros(0, dtype=bool)
        nx, ny, nz = self.phi.shape[:3]
        lo = self.origin + 0.5 * self.pitch
        hi = self.origin + (np.array([nx, ny, nz]) - 0.5) * self.pitch

        best_p = q.copy()
        best_res = np.full(N, np.inf)
        diag3 = np.arange(3)

        def _sweep(seed: np.ndarray, idx: np.ndarray, n_iter: int) -> None:
            """Adaptive-LM refine the points `idx`, warm-started at seed[idx].
            Updates best_p / best_res in place."""
            if idx.size == 0:
                return
            p = seed[idx].copy()
            res = _sample_vec(self.phi, self.origin, self.pitch, p) - q[idx]
            rn = np.linalg.norm(res, axis=1)
            better = rn < best_res[idx]
            best_res[idx[better]] = rn[better]
            best_p[idx[better]] = p[better]
            lam = np.full(idx.size, 1e-3)
            act = np.where(rn >= tol)[0]              # local indices into idx
            for _ in range(n_iter):
                if act.size == 0:
                    break
                J = _sample_jac(self.jac, self.origin, self.pitch, p[act])  # (n,3,3)
                Jt = np.transpose(J, (0, 2, 1))
                JtJ = Jt @ J
                g = (Jt @ res[act][:, :, None])[:, :, 0]                    # (n,3)
                A = JtJ.copy()
                # Marquardt damping (scale by the diagonal) + a tiny absolute
                # floor so A stays positive-definite even when J is singular
                # (a column of J vanishes at a fold) → solve never raises.
                A[:, diag3, diag3] += lam[act, None] * JtJ[:, diag3, diag3] + 1e-9
                dp = np.linalg.solve(A, g[:, :, None])[:, :, 0]
                p_try = np.clip(p[act] - dp, lo, hi)
                res_try = _sample_vec(self.phi, self.origin, self.pitch, p_try) - q[idx[act]]
                rn_try = np.linalg.norm(res_try, axis=1)

                acc = rn_try < rn[act]
                ga = act[acc]                          # accepted → take step, relax λ
                p[ga] = p_try[acc]
                res[ga] = res_try[acc]
                rn[ga] = rn_try[acc]
                lam[ga] = np.maximum(lam[ga] * 0.3, 1e-7)
                rj = act[~acc]                         # rejected → stay put, damp more
                lam[rj] = np.minimum(lam[rj] * 3.0, 1e7)

                imp = rn[ga] < best_res[idx[ga]]
                gi = ga[imp]
                best_res[idx[gi]] = rn[gi]
                best_p[idx[gi]] = p[gi]
                act = act[rn[act] >= tol]              # drop converged

        _sweep(q, np.arange(N), iters)

        # Second chance for stragglers: re-seed each still-unconverged point from
        # the nearest CONVERGED point in array order (consecutive toolpath points
        # are spatially close, so a neighbour's original-space position is a much
        # better warm start than q at a fold) and refine again. Only the leftover
        # tips re-run, so this is cheap.
        bad = np.where(best_res >= tol)[0]
        if bad.size and bad.size < N:
            seed = _nearest_good_fill(best_p, best_res < tol)
            _sweep(seed, bad, max(8, iters // 2))

        converged = best_res < tol
        return best_p, converged


def solve_deformation_map(
    growth: GrowthResult,
    *,
    displacement_smooth_sigma: float = 2.0,
    max_tilt_deg: float = 30.0,
    bed_blend_height: float = 2.0,
    eps: float = 1e-6,
    rtol: float = 1e-7,
    maxiter: int = 5000,
) -> DeformationMap:
    """Solve the 3-coordinate Poisson system for Φ on the voxel grid.

    See the module docstring for the formulation. Returns a DeformationMap
    with Φ extended to the whole grid (outside-model cells carry the nearest
    in-model displacement, so Φ is continuous across the surface) and its
    trilinear-gradient Jacobian fields precomputed for the Newton inverse.

    `displacement_smooth_sigma` (voxels) Gaussian-smooths the solved
    displacement field so Φ becomes ~C1 instead of C0-trilinear — removing the
    voxel-scale surface waviness AND making the map globally injective (no
    folded tips). See the smoothing block below for measured effect. NOTE: it
    changes the deformation, so the export that gets sliced and the inverse
    that undoes it must use the SAME value (like dz for the Z-only path).

    The rotations are driven by the local BFS growth vectors (lowest
    distortion). Must match between the sliced export and the inverse.
    """
    matrix = growth.step >= 0
    nx, ny, nz = matrix.shape
    pitch = float(growth.pitch)
    origin = np.asarray(growth.origin, dtype=np.float64)

    # Regular original-grid node positions (cell centres).
    iidx = np.arange(nx)
    jidx = np.arange(ny)
    kidx = np.arange(nz)
    # P_orig[i,j,k] = origin + (idx + 0.5)*pitch
    P = np.stack(np.meshgrid(iidx, jidx, kidx, indexing="ij"), axis=-1).astype(np.float64)
    P = origin[None, None, None, :] + (P + 0.5) * pitch  # (nx,ny,nz,3)

    n_model = int(matrix.sum())
    if n_model == 0:
        phi = P.copy()
        jac = np.broadcast_to(np.eye(3), (nx, ny, nz, 3, 3)).copy()
        return DeformationMap(phi=phi, jac=jac, origin=origin, pitch=pitch)

    order = np.argwhere(matrix).astype(np.int64)         # (M,3)
    flat_idx = np.full(matrix.shape, -1, dtype=np.int64)
    flat_idx[matrix] = np.arange(n_model)

    # Per-model-voxel rotation taking the local BFS growth dir → +ẑ.
    vecs = growth.vectors[matrix]
    R = rotations_to_vertical(vecs)                       # (M,3,3)

    rows_l, cols_l, vals_l = [], [], []
    b = np.zeros((n_model, 3), dtype=np.float64)          # RHS, one column per coord
    for off in _CANON_OFFSETS:
        d_world = off.astype(np.float64) * pitch          # original edge offset (3,)
        nbr = order + off
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
        i_arr = src[valid]
        j_arr = jj[valid]
        # Target offset Φ_j − Φ_i = R̄_ij d_world, R̄ = ½(R_i + R_j).
        Rbar = 0.5 * (R[i_arr] + R[j_arr])               # (E,3,3)
        t = Rbar @ d_world                                # (E,3)
        ones = np.ones(i_arr.size, dtype=np.float64)
        rows_l += [i_arr, j_arr, i_arr, j_arr]
        cols_l += [i_arr, j_arr, j_arr, i_arr]
        vals_l += [ones, ones, -ones, -ones]
        np.add.at(b, i_arr, -t)
        np.add.at(b, j_arr, t)

    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    lap = coo_array((vals, (rows, cols)), shape=(n_model, n_model)).tocsr()

    Pm = P[matrix]                                        # (M,3) original world pos

    # Solve for the DISPLACEMENT U = Φ − P, pinned to 0 at the bed — NOT Φ
    # directly. Pinning Φ to the absolute bed position (~125 mm) via a 1e6×
    # penalty would push the RHS to ~big·125 ≈ 1e9, so CG's relative tolerance
    # becomes a ~100 mm absolute one and it "converges" (info=0) to garbage.
    # The displacement RHS  b_u = b − L·P  is the divergence of (R̄−I)d — it is
    # exactly 0 where the rotations are identity (flat-bottom regions) and O(1)
    # elsewhere, so CG stays well-scaled. (This mirrors the scalar
    # integrate_vectors_to_potential, which pins φ→0 for the same reason.)
    b_u = b - lap @ Pm                                    # (M,3)

    # Pin bed seeds (lowest non-empty Z layer) to U = 0 (bed stays put).
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
    # seeds: penalty drags U_seed → 0; RHS there stays small.

    u_model = np.zeros((n_model, 3), dtype=np.float64)
    for c in range(3):
        try:
            sol, info = cg(a_mat, b_u[:, c], rtol=rtol, maxiter=maxiter)
        except TypeError:  # SciPy < 1.12
            sol, info = cg(a_mat, b_u[:, c], tol=rtol, maxiter=maxiter)
        if info != 0:
            sol = spsolve(a_mat, b_u[:, c])
        u_model[:, c] = sol
    phi_model = Pm + u_model

    # Scatter solution back; extend the DISPLACEMENT field outward so Φ is
    # continuous across the model boundary (gcode travels / surface vertices
    # sample just outside the solid).
    phi = P.copy()
    phi[matrix] = phi_model
    U = phi - P                                           # displacement field
    outside = ~matrix
    if outside.any():
        idx = distance_transform_edt(outside, return_distances=False, return_indices=True)
        U[outside] = U[idx[0], idx[1], idx[2]][outside]

    # Gaussian smoothing of the displacement field tames the sharp per-voxel
    # inconsistencies that fold the map (non-injective tips → Newton spikes) and
    # the voxel-scale surface facets. Default 2.0 = smoother mesh, fewer folds.
    # It DOES soften real curvature (a bore-ceiling dome shrinks ~1.5→0.7 mm at
    # sigma=2); the dome still prints if you also pass --subdivide-error so the
    # mesh can represent it. Drop sigma toward 0.5 for a sharper/bigger dome at
    # the cost of more surface texture. MUST match between the sliced export and
    # the inverse (it changes Φ).
    if displacement_smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        for c in range(3):
            U[..., c] = gaussian_filter(U[..., c], displacement_smooth_sigma, mode="nearest")

    # Bed blend: ramp the displacement to ZERO at the plate so the first layers
    # stay flat at their sliced height. Without it the deformation already tilts
    # just above the pinned bed seeds (growth tilts under blades), so the
    # inverse maps the sliced first layer to a varying — partly NEGATIVE —
    # original Z and the nozzle digs into the bed. Identity at z=bed, full
    # deformation above bed_blend_height (mirrors the Z-only path's bed_blend).
    if bed_blend_height and bed_blend_height > 0:
        bed_z = float(P[matrix][:, 2].min())
        w = np.clip((P[:, :, :, 2] - bed_z) / bed_blend_height, 0.0, 1.0)
        U *= w[..., None]
    phi = P + U

    # Trilinear-gradient Jacobian fields for the Newton inverse. np.gradient
    # gives ∂phi_c/∂(grid index); divide by pitch for ∂/∂(world coord).
    jac = np.empty((nx, ny, nz, 3, 3), dtype=np.float64)
    for c in range(3):
        gx, gy, gz = np.gradient(phi[..., c])
        jac[..., c, 0] = gx / pitch
        jac[..., c, 1] = gy / pitch
        jac[..., c, 2] = gz / pitch

    return DeformationMap(phi=phi, jac=jac, origin=origin, pitch=pitch)


def _face_nonaffinity(mesh: trimesh.Trimesh, dmap: DeformationMap) -> np.ndarray:
    """Per-face deformation error: ‖Φ(centroid) − mean(Φ(verts))‖. Nonzero where
    Φ curves across the face — i.e. where the face is too coarse to represent the
    deformation (a big flat triangle stays flat instead of bowing)."""
    V = mesh.vertices.astype(np.float64, copy=False)
    F = mesh.faces
    cen = V[F].mean(axis=1)
    phi_v = dmap.forward_points(V)
    return np.linalg.norm(dmap.forward_points(cen) - phi_v[F].mean(axis=1), axis=1)


def _split_tris(a, b, c, mab, mbc, mca):
    """Sub-triangles of (a,b,c) for a longest-edge bisection, given the midpoint
    vertex id of each edge (-1 = that edge isn't being split). The triangle is
    pre-rotated so (a,b) is the LONGEST edge, and the conformity invariant holds:
    if any edge is split, the longest one is too — so `mab` is always set when
    any midpoint is. Bisecting the longest edge first (m_ab→opposite vertex),
    then each further marked edge, keeps Rivara's bounded-angle property and,
    because edge midpoints are shared between neighbours, stays crack-free."""
    if mab < 0:                                   # nothing marked
        return ((a, b, c),)
    if mbc < 0 and mca < 0:                        # longest only → 2
        return ((a, mab, c), (mab, b, c))
    if mca < 0:                                    # longest + bc → 3
        return ((a, mab, c), (mab, b, mbc), (mab, mbc, c))
    if mbc < 0:                                    # longest + ca → 3
        return ((a, mab, mca), (mab, c, mca), (mab, b, c))
    return (                                        # all three → 4
        (a, mab, mca), (mab, c, mca), (mab, b, mbc), (mab, mbc, c))


def _adaptive_refine(
    mesh: trimesh.Trimesh, dmap: DeformationMap, max_error: float,
    max_faces: int, max_passes: int = 12,
) -> trimesh.Trimesh:
    """Conforming adaptive remesh by **Rivara longest-edge bisection**: refine
    only the faces whose Φ-non-affinity (see _face_nonaffinity) exceeds
    `max_error`, by bisecting longest edges, until none are left or `max_faces`
    is hit. Crack-free without uniform subdivision's 4×-per-pass blowup.

    Each pass: mark the longest edge of every over-error face, then take the
    **longest-edge closure** — if any edge of a face is marked, mark its longest
    edge too (iterated to a fixed point). Because an edge is shared by its two
    faces, marking is symmetric, so both faces split it at the SAME midpoint →
    no T-junctions. Faces are then split per their marked-edge pattern."""
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    for _ in range(max_passes):
        if F.shape[0] >= max_faces:
            break
        cen = V[F].mean(axis=1)
        phiV = dmap.forward_points(V)
        err = np.linalg.norm(dmap.forward_points(cen) - phiV[F].mean(axis=1), axis=1)
        bad = err > max_error
        if not bad.any():
            break

        n = F.shape[0]
        # Unique undirected edges + per-face local→global edge ids.
        loc = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)  # (n,3,2)
        ek = np.sort(loc, axis=2).reshape(-1, 2)
        uniq, inv = np.unique(ek, axis=0, return_inverse=True)
        face_edges = inv.ravel().reshape(n, 3)
        elen = np.linalg.norm(V[uniq[:, 0]] - V[uniq[:, 1]], axis=1)
        longest_local = np.argmax(elen[face_edges], axis=1)                 # (n,)
        face_long_eid = face_edges[np.arange(n), longest_local]

        # Mark + longest-edge closure.
        marked = np.zeros(uniq.shape[0], dtype=bool)
        marked[face_long_eid[bad]] = True
        while True:
            need = marked[face_edges].any(axis=1) & ~marked[face_long_eid]
            if not need.any():
                break
            marked[face_long_eid[need]] = True

        # One midpoint vertex per marked edge (shared → conforming).
        midids = np.full(uniq.shape[0], -1, dtype=np.int64)
        me = np.where(marked)[0]
        if me.size == 0:
            break
        midids[me] = V.shape[0] + np.arange(me.size)
        V = np.vstack([V, 0.5 * (V[uniq[me, 0]] + V[uniq[me, 1]])])

        # Split each touched face; untouched faces pass through as a block.
        nm = marked[face_edges].sum(axis=1)
        out = [F[nm == 0]]
        changed = np.where(nm > 0)[0]
        rows: list[tuple[int, int, int]] = []
        for t in changed:
            s = int(longest_local[t])
            o0, o1, o2 = s, (s + 1) % 3, (s + 2) % 3
            rows.extend(_split_tris(
                int(F[t, o0]), int(F[t, o1]), int(F[t, o2]),
                int(midids[face_edges[t, o0]]),
                int(midids[face_edges[t, o1]]),
                int(midids[face_edges[t, o2]]),
            ))
        if rows:
            out.append(np.asarray(rows, dtype=np.int64))
        F = np.vstack([b for b in out if b.size])
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def deform_mesh_3d(
    mesh: trimesh.Trimesh,
    dmap: DeformationMap,
    subdivide_max_error: float = 0.0,
    max_faces: int = 2_000_000,
    refine: str = "adaptive",
) -> trimesh.Trimesh:
    """Apply the full-3D map Φ to every mesh vertex (all axes move).

    The map is applied PER VERTEX, so a flat region with few/large triangles
    can't follow Φ's curvature — it stays flat (e.g. a bore ceiling that should
    bow). `subdivide_max_error` (mm) > 0 refines the mesh until the worst
    per-face non-affinity (see _face_nonaffinity) drops below it, capped at
    `max_faces`:

    - `refine="adaptive"` (default): **Rivara longest-edge bisection**
      (`_adaptive_refine`) — conforming/crack-free and refines ONLY the curved
      faces, so it reaches the same quality at far fewer faces than uniform
      (propeller ~15× fewer).
    - `refine="uniform"`: trimesh's 1→4 subdivide of EVERY face each pass. Simple
      and watertight but blows the face count up; kept as a fallback."""
    cur = mesh
    if subdivide_max_error and subdivide_max_error > 0:
        if refine == "uniform":
            while True:
                err = _face_nonaffinity(cur, dmap)
                if err.size == 0 or err.max() <= subdivide_max_error:
                    break
                if len(cur.faces) * 4 > max_faces:
                    break
                cur = cur.subdivide()
        else:
            cur = _adaptive_refine(cur, dmap, subdivide_max_error, max_faces)
    new_verts = dmap.forward_points(cur.vertices.astype(np.float64, copy=False))
    return trimesh.Trimesh(vertices=new_verts, faces=cur.faces, process=False)


def _despike_path(p: np.ndarray, thresh: float = 1.5) -> int:
    """In-place repair of isolated "out-and-back" spikes in an ordered point
    path. A spike is one point far from BOTH array-neighbours while the two
    neighbours are close to each other — the signature of a wrong-branch
    inverse at a fold (the toolpath darts to the grid ceiling and back).
    A travel move is a single jump, not out-and-back, so it never matches.

    Returns the number of points repaired (replaced by the neighbour midpoint).
    """
    n = p.shape[0]
    if n < 3:
        return 0
    d_prev = np.linalg.norm(p[1:-1] - p[:-2], axis=1)
    d_next = np.linalg.norm(p[1:-1] - p[2:], axis=1)
    d_chord = np.linalg.norm(p[2:] - p[:-2], axis=1)
    spike = (d_prev > thresh) & (d_next > thresh) & (d_chord < thresh)
    idx = np.where(spike)[0] + 1  # back to absolute indices
    if idx.size:
        p[idx] = 0.5 * (p[idx - 1] + p[idx + 1])
    return int(idx.size)


class BackTransform3D:
    """Full-3D map adapter exposing the same forward_points_batch /
    invert_points_batch interface as backtransform.BackTransform, so it drops
    straight into backtransform_gcode_file. Unlike the Z-only BackTransform
    this moves XY too (it wraps a DeformationMap, not a 1-D depth column)."""

    is_3d = True  # marker: the gcode driver must not use the depth-field MP path

    def __init__(self, dmap: DeformationMap, mesh: trimesh.Trimesh | None = None) -> None:
        self.dmap = dmap
        # The ORIGINAL placed mesh (same coords the inverse maps back into).
        # Kept so the post-inverse overhang/bridge cooling pass can ask "is
        # there part material directly below this toolpath point?". Optional —
        # None when the map was built without a source mesh (tests).
        self.mesh = mesh

    @classmethod
    def from_mesh(
        cls,
        stl_path: str,
        *,
        volume_side: float = 250.0,
        pitch: float = 1.0,
        max_tilt_deg: float = 30.0,
        smooth_sigma: float = 2.0,
        xy_center: tuple[float, float] | None = None,
    ) -> "BackTransform3D":
        """Voxelise + grow + solve the deformation map for `stl_path`, placed
        the same way the Z-only path places it (Z_min→0, XY centred on
        `xy_center` if given, else the build-volume centre). `smooth_sigma`
        and `max_tilt_deg` must match the values the sliced export was built
        with."""
        from .build_volume import BuildVolume
        from .growth import compute_growth
        from .mesh_io import load_and_place
        from .voxelize import voxelize_solid

        mesh = trimesh.load(stl_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Not a single mesh: {type(mesh).__name__}")
        if xy_center is not None:
            c = 0.5 * (mesh.bounds[0, :2] + mesh.bounds[1, :2])
            mesh.apply_translation([
                float(xy_center[0]) - float(c[0]),
                float(xy_center[1]) - float(c[1]),
                -float(mesh.bounds[0, 2]),
            ])
        else:
            mesh = load_and_place(stl_path, BuildVolume.cube(volume_side))
        vg = voxelize_solid(mesh, pitch=pitch)
        gr = compute_growth(vg, max_tilt_deg=max_tilt_deg)
        dmap = solve_deformation_map(
            gr, displacement_smooth_sigma=smooth_sigma, max_tilt_deg=max_tilt_deg,
        )
        return cls(dmap, mesh=mesh)

    def forward_points_batch(self, xyz_orig: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz_orig, dtype=np.float64)
        if xyz.shape[0] == 0:
            return xyz.copy()
        return self.dmap.forward_points(xyz)

    def invert_points_batch(self, xyz_def: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz_def, dtype=np.float64)
        if xyz.shape[0] == 0:
            return xyz.copy()
        p, conv = self.dmap.inverse_points(xyz)
        n_bad = int((~conv).sum())
        n_spikes = _despike_path(p, thresh=1.5)
        if n_bad:
            import sys
            print(
                f"  [3d-inverse] {n_bad:,}/{len(conv):,} points did not converge "
                "(folded / overhang region where Φ is not injective); "
                f"{n_spikes:,} isolated wrong-branch spikes repaired by neighbour "
                "interpolation. Tips remain the unreliable region.",
                file=sys.stderr, flush=True,
            )
        return p

    def overhang_degree(
        self, xyz_orig: np.ndarray, probe: float = 0.8, min_z: float = 0.6,
        n_probes: int = 4,
    ) -> np.ndarray:
        """Per-point overhang SEVERITY in [0, 1] for the post-inverse cooling
        ramp: 0 = fully supported (part material the whole way down the probe
        column), 1 = fully unsupported (a bridge — air all `probe` mm below).

        The slicer scheduled cooling (M106) from the DEFORMED, flat geometry,
        where every layer rests squarely on the one beneath it. After the
        inverse maps the path back onto the curved part, a segment can hang over
        a void the slicer never saw. On this 3-axis machine the nozzle is
        vertical, so material is laid on whatever is directly below in world Z —
        hence "support" ⇔ part material straight down. We sample `n_probes`
        evenly spaced depths in (0, probe] and return the FRACTION that fall
        OUTSIDE the part (`contains` False). That fraction is the continuous
        analogue of a slicer's overhang/overlap % (overlap ≈ 1 − degree), so the
        caller can ramp the fan with severity instead of switching it fully on:
        a near-vertical wall has solid right below (→ ~0), a steep overhang sits
        out over more air (→ mid), a flat bridge has none (→ 1). Walls / infill
        / top-surfaces read ~0. Points within `min_z` of the plate are forced to
        0 (the bed supports them). All-zero if no mesh was stored."""
        xyz = np.asarray(xyz_orig, dtype=np.float64)
        n = xyz.shape[0]
        if self.mesh is None or n == 0 or n_probes < 1:
            return np.zeros(n, dtype=np.float64)
        bed_z = float(self.mesh.bounds[0, 2])
        air = np.zeros(n, dtype=np.float64)
        for k in range(n_probes):
            depth = (k + 0.5) / n_probes * float(probe)
            pp = xyz.copy()
            pp[:, 2] -= depth
            air += ~np.asarray(self.mesh.contains(pp), dtype=bool)
        degree = air / n_probes
        degree[xyz[:, 2] <= bed_z + float(min_z)] = 0.0
        return degree


# --------------------------------------------------------------------------
# Trilinear samplers (vector field + Jacobian field). Out-of-bounds clamps to
# the boundary cell, matching deform._sample_trilinear's convention.
# --------------------------------------------------------------------------
def _tri_weights(field_shape, origin, pitch, points):
    nx, ny, nz = field_shape[:3]
    p = (points - origin) / pitch - 0.5
    p[:, 0] = np.clip(p[:, 0], 0.0, nx - 1.0001)
    p[:, 1] = np.clip(p[:, 1], 0.0, ny - 1.0001)
    p[:, 2] = np.clip(p[:, 2], 0.0, nz - 1.0001)
    i0 = np.floor(p[:, 0]).astype(np.int64); fx = p[:, 0] - i0
    j0 = np.floor(p[:, 1]).astype(np.int64); fy = p[:, 1] - j0
    k0 = np.floor(p[:, 2]).astype(np.int64); fz = p[:, 2] - k0
    return i0, j0, k0, fx, fy, fz


def _sample_vec(field, origin, pitch, points):
    """Trilinear sample of an (nx,ny,nz,3) field at world points → (N,3)."""
    i0, j0, k0, fx, fy, fz = _tri_weights(field.shape, origin, pitch, points)
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    fx = fx[:, None]; fy = fy[:, None]; fz = fz[:, None]
    c000 = field[i0, j0, k0]; c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]; c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]; c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]; c111 = field[i1, j1, k1]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def _sample_jac(field, origin, pitch, points):
    """Trilinear sample of an (nx,ny,nz,3,3) field → (N,3,3)."""
    i0, j0, k0, fx, fy, fz = _tri_weights(field.shape, origin, pitch, points)
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    fx = fx[:, None, None]; fy = fy[:, None, None]; fz = fz[:, None, None]
    c000 = field[i0, j0, k0]; c100 = field[i1, j0, k0]
    c010 = field[i0, j1, k0]; c110 = field[i1, j1, k0]
    c001 = field[i0, j0, k1]; c101 = field[i1, j0, k1]
    c011 = field[i0, j1, k1]; c111 = field[i1, j1, k1]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz
