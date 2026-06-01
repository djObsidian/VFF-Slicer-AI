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

    # ---- inverse: deformed → original (vectorised Newton) ----
    def inverse_points(
        self, q: np.ndarray, iters: int = 20, tol: float = 1e-4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (p, converged_mask). p[k] satisfies Φ(p[k]) ≈ q[k].

        Newton step:  p ← p − JΦ(p)^{-1} (Φ(p) − q).  Warm-started at q
        (XY deformation is usually modest). `converged_mask` is False where
        the iteration didn't reach `tol` (typically folded regions)."""
        q = np.asarray(q, dtype=np.float64)
        N = q.shape[0]
        if N == 0:
            return q.copy(), np.zeros(0, dtype=bool)
        p = q.copy()
        nx, ny, nz = self.phi.shape[:3]
        lo = self.origin + 0.5 * self.pitch
        hi = self.origin + (np.array([nx, ny, nz]) - 0.5) * self.pitch
        res = np.full(N, np.inf)
        active = np.arange(N)  # indices still being iterated
        for _ in range(iters):
            if active.size == 0:
                break
            pa = p[active]
            r = _sample_vec(self.phi, self.origin, self.pitch, pa) - q[active]
            rn = np.linalg.norm(r, axis=1)
            res[active] = rn
            # Drop converged points from the active set — most converge in a few
            # iterations, so this stops us re-sampling the whole array every step.
            keep = rn >= tol
            if not keep.any():
                active = active[:0]
                break
            work = active[keep]
            rw = r[keep]
            J = _sample_jac(self.jac, self.origin, self.pitch, p[work])  # (n,3,3)
            # Solve n independent 3×3 systems J·dp = rw. Freeze near-singular
            # rows (folded regions) so one bad cell can't NaN the whole batch.
            dp = np.zeros_like(rw)
            detok = np.abs(np.linalg.det(J)) > 1e-9
            if detok.any():
                dp[detok] = np.linalg.solve(J[detok], rw[detok][:, :, None])[:, :, 0]
            p[work] = np.clip(p[work] - dp, lo, hi)
            active = work
        converged = res < tol
        return p, converged


def solve_deformation_map(
    growth: GrowthResult,
    *,
    eps: float = 1e-6,
    rtol: float = 1e-7,
    maxiter: int = 5000,
) -> DeformationMap:
    """Solve the 3-coordinate Poisson system for Φ on the voxel grid.

    See the module docstring for the formulation. Returns a DeformationMap
    with Φ extended to the whole grid (outside-model cells carry the nearest
    in-model displacement, so Φ is continuous across the surface) and its
    trilinear-gradient Jacobian fields precomputed for the Newton inverse.
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

    # Per-model-voxel rotation taking growth dir → +ẑ.
    R = rotations_to_vertical(growth.vectors[matrix])     # (M,3,3)

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


def deform_mesh_3d(mesh: trimesh.Trimesh, dmap: DeformationMap) -> trimesh.Trimesh:
    """Apply the full-3D map Φ to every mesh vertex (all axes move)."""
    new_verts = dmap.forward_points(mesh.vertices.astype(np.float64, copy=False))
    return trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)


class BackTransform3D:
    """Full-3D map adapter exposing the same forward_points_batch /
    invert_points_batch interface as backtransform.BackTransform, so it drops
    straight into backtransform_gcode_file. Unlike the Z-only BackTransform
    this moves XY too (it wraps a DeformationMap, not a 1-D depth column)."""

    is_3d = True  # marker: the gcode driver must not use the depth-field MP path

    def __init__(self, dmap: DeformationMap) -> None:
        self.dmap = dmap

    @classmethod
    def from_mesh(
        cls,
        stl_path: str,
        *,
        volume_side: float = 250.0,
        pitch: float = 1.0,
        max_tilt_deg: float = 30.0,
        xy_center: tuple[float, float] | None = None,
    ) -> "BackTransform3D":
        """Voxelise + grow + solve the deformation map for `stl_path`, placed
        the same way the Z-only path places it (Z_min→0, XY centred on
        `xy_center` if given, else the build-volume centre)."""
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
        return cls(solve_deformation_map(gr))

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
        if n_bad:
            import sys
            print(
                f"  [3d-inverse] WARNING: {n_bad:,}/{len(conv):,} points did not "
                "converge (folded / overhang region where Φ is not injective) — "
                "left at best-effort position. These are the unreliable tips.",
                file=sys.stderr, flush=True,
            )
        return p


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
