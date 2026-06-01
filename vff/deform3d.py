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
        this ignores the in-plane width change a 3-axis nozzle can't realize."""
        J = _sample_jac(self.jac, self.origin, self.pitch, np.asarray(pts, float))
        return np.linalg.inv(J)[:, 2, 2]

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
        # Track the BEST iterate per point, not the last. A non-convergent point
        # (folded / overhang region) otherwise drifts to the grid-clamp boundary
        # and returns Z = grid-top garbage — that's what produced the wedge
        # spikes to the ceiling in the gcode. The lowest-residual iterate is the
        # closest sane approximation instead.
        best_p = q.copy()
        best_res = np.full(N, np.inf)
        active = np.arange(N)  # indices still being iterated
        for _ in range(iters):
            if active.size == 0:
                break
            pa = p[active]
            r = _sample_vec(self.phi, self.origin, self.pitch, pa) - q[active]
            rn = np.linalg.norm(r, axis=1)
            improved = rn < best_res[active]
            ai = active[improved]
            best_res[ai] = rn[improved]
            best_p[ai] = pa[improved]
            # Drop converged points from the active set — most converge in a few
            # iterations, so this stops us re-sampling the whole array every step.
            keep = rn >= tol
            if not keep.any():
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
        converged = best_res < tol
        return best_p, converged


def _geodesic_direction_field(growth: GrowthResult, sigma: float, max_tilt_deg: float) -> np.ndarray:
    """Direction field from the gradient of the geodesic (FMM/Dijkstra) depth.

    The local BFS growth vectors only know where each voxel was *reached from*,
    so above a hole (e.g. a bore ceiling) they point ~straight up and the
    deformation under-domes that region. The geodesic depth, by contrast,
    encodes the detour the front had to take *around* the hole — its gradient
    tilts toward the deeper centre, so aligning it to vertical actually domes
    the ceiling. Normalized and tilt-clamped like the BFS field."""
    from .deform import smoothed_depth_field
    from .growth import clamp_to_vertical
    phi = smoothed_depth_field(growth, sigma=sigma, method="fmm", outside_mode="extend")
    gx, gy, gz = np.gradient(phi)
    V = np.stack([gx, gy, gz], axis=-1).astype(np.float32)
    n = np.linalg.norm(V, axis=-1, keepdims=True)
    V = np.where(n > 1e-9, V / n, 0.0).astype(np.float32)
    V[growth.step < 0] = 0.0
    return clamp_to_vertical(V, max_tilt_deg)


def solve_deformation_map(
    growth: GrowthResult,
    *,
    displacement_smooth_sigma: float = 2.0,
    growth_source: str = "bfs",
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

    `growth_source`: 'bfs' (default) drives the rotations from the local BFS
    growth vectors — lowest distortion. 'geodesic' drives them from the geodesic
    depth gradient, which captures detours around holes (domes a bore ceiling
    that 'bfs' leaves nearly flat) at a modest global distortion cost. Must
    match between the sliced export and the inverse.
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

    # Per-model-voxel rotation taking growth dir → +ẑ. 'geodesic' swaps the
    # local BFS direction for the geodesic-depth gradient (domes ceilings).
    if growth_source == "geodesic":
        dir_field = _geodesic_direction_field(growth, displacement_smooth_sigma, max_tilt_deg)
        vecs = dir_field[matrix]
    else:
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


def deform_mesh_3d(
    mesh: trimesh.Trimesh,
    dmap: DeformationMap,
    subdivide_max_error: float = 0.0,
    max_faces: int = 2_000_000,
) -> trimesh.Trimesh:
    """Apply the full-3D map Φ to every mesh vertex (all axes move).

    The map is applied PER VERTEX, so a flat region with few/large triangles
    can't follow Φ's curvature — it stays flat (e.g. a bore ceiling that should
    bow). `subdivide_max_error` (mm) > 0 uniformly subdivides the mesh until the
    worst per-face non-affinity (see _face_nonaffinity) drops below it, capped at
    `max_faces`. Uniform subdivision is used because it stays watertight (no
    T-junction cracks); it's heavier than a conforming adaptive remesh (Rivara
    longest-edge bisection would hit the same quality at ~15× fewer faces — a
    backlog item), but safe for slicing."""
    cur = mesh
    if subdivide_max_error and subdivide_max_error > 0:
        while True:
            err = _face_nonaffinity(cur, dmap)
            if err.size == 0 or err.max() <= subdivide_max_error:
                break
            if len(cur.faces) * 4 > max_faces:
                break
            cur = cur.subdivide()
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
        smooth_sigma: float = 2.0,
        growth_source: str = "bfs",
        xy_center: tuple[float, float] | None = None,
    ) -> "BackTransform3D":
        """Voxelise + grow + solve the deformation map for `stl_path`, placed
        the same way the Z-only path places it (Z_min→0, XY centred on
        `xy_center` if given, else the build-volume centre). `smooth_sigma`,
        `growth_source` and `max_tilt_deg` must match the values the sliced
        export was built with."""
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
        return cls(solve_deformation_map(
            gr, displacement_smooth_sigma=smooth_sigma,
            growth_source=growth_source, max_tilt_deg=max_tilt_deg,
        ))

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
