"""Spatial growth algorithm: BFS from the printer-bed surface upward through
a solid-voxelized model.

For each voxel we produce:

  - step index (0 = bed seed, 1..N = BFS layer, -1 = outside model)
  - a non-unit vector pointing from the older "source" voxel toward this
    voxel — i.e. the local *growth direction*.

How the vector is computed (one BFS iteration):

  1. BFS expansion uses **26-connectivity** (face + edge + vertex neighbours),
     so a voxel can be reached by a diagonal predecessor when no face
     predecessor exists. This is what produces the non-trivial √2 / √3
     magnitudes the user wants — pure 6-connectivity always gives a face
     predecessor (distance 1) for every voxel.

  2. For each newly-painted voxel V we look at its 26 neighbours and find
     all that are *older* (step < V's step). We classify them by squared
     Euclidean distance: 1 (face), 2 (edge), 3 (vertex).

  3. We keep ONLY the smallest non-empty class — that's the "nearest older"
     set. Within that class we average the offset vectors (handles ties:
     several equidistant predecessors get blended), normalize the average,
     and scale it to the canonical length √(class) so the resulting
     magnitude is exactly 1, √2, or √3.

  4. The vector is then **negated**, so it points from older → newer
     (the growth direction the visualization shows).

Tie-handling note: averaging within the nearest class is a deliberate
choice — it gives a smoother field than picking one canonical neighbour,
and the magnitude still snaps to 1 / √2 / √3 exactly. The literal "single
nearest with canonical tie-break" variant is sketched in the backlog if
we ever want stricter semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .voxelize import VoxelGrid


# Offsets partitioned by squared Euclidean distance.
_OFFSETS_FACE = np.array(
    [(1, 0, 0), (-1, 0, 0),
     (0, 1, 0), (0, -1, 0),
     (0, 0, 1), (0, 0, -1)],
    dtype=np.int32,
)  # d² = 1

_OFFSETS_EDGE = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if dx * dx + dy * dy + dz * dz == 2],
    dtype=np.int32,
)  # d² = 2, 12 offsets

_OFFSETS_VERTEX = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if dx * dx + dy * dy + dz * dz == 3],
    dtype=np.int32,
)  # d² = 3, 8 offsets


@dataclass
class GrowthResult:
    step: np.ndarray       # int32 (nx, ny, nz). -1 = outside model; 0 = bed seed; 1..n_steps-1 = BFS layer.
    vectors: np.ndarray    # float32 (nx, ny, nz, 3). Growth direction (older -> newer); magnitude in {1, √2, √3}. Zero for step 0 and outside.
    n_steps: int           # max(step) + 1
    pitch: float
    origin: np.ndarray     # (3,) corner of voxel (0,0,0) in world coords (mirrors VoxelGrid.origin)
    connectivity: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.step.shape  # type: ignore[return-value]

    @property
    def painted_count(self) -> int:
        return int((self.step >= 0).sum())


def _seed_bed_mask(matrix: np.ndarray) -> np.ndarray:
    """Voxels in the lowest non-empty Z layer of the model become seeds."""
    nz = matrix.shape[2]
    seed = np.zeros_like(matrix)
    for k in range(nz):
        if matrix[:, :, k].any():
            seed[:, :, k] = matrix[:, :, k]
            return seed
    return seed  # empty model


def _dilate26(mask: np.ndarray) -> np.ndarray:
    """26-connectivity binary dilation, pure numpy.

    Shifts the mask by every (dx,dy,dz) in {-1,0,1}^3 \\ {(0,0,0)} and ORs
    back into the result. Each shift is one vectorized array OR — total of
    26 ops, all in C — fast enough that we don't need scipy.ndimage."""
    out = mask.copy()
    nx, ny, nz = mask.shape
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                sx_dst = slice(max(dx, 0), nx + min(dx, 0))
                sx_src = slice(max(-dx, 0), nx + min(-dx, 0))
                sy_dst = slice(max(dy, 0), ny + min(dy, 0))
                sy_src = slice(max(-dy, 0), ny + min(-dy, 0))
                sz_dst = slice(max(dz, 0), nz + min(dz, 0))
                sz_src = slice(max(-dz, 0), nz + min(-dz, 0))
                out[sx_dst, sy_dst, sz_dst] |= mask[sx_src, sy_src, sz_src]
    return out


def _accumulate_offsets(
    new_idx: np.ndarray,         # (N, 3) int32 — indices of the new frontier
    offsets: np.ndarray,         # (K, 3) int32 — neighbour offsets to test
    step_arr: np.ndarray,        # (nx,ny,nz) int32 — current step state
    next_step: int,              # step we're painting in this iteration
    shape_arr: np.ndarray,       # (3,) int32 — grid shape
) -> tuple[np.ndarray, np.ndarray]:
    """For each new voxel, sum offset vectors to older neighbours (step < next_step)
    among the given offsets. Returns (count[N] int32, sum_dir[N,3] float32)."""
    n = len(new_idx)
    count = np.zeros(n, dtype=np.int32)
    sum_dir = np.zeros((n, 3), dtype=np.float32)
    for d in offsets:
        nbr = new_idx + d
        inb = (
            (nbr[:, 0] >= 0) & (nbr[:, 0] < shape_arr[0])
            & (nbr[:, 1] >= 0) & (nbr[:, 1] < shape_arr[1])
            & (nbr[:, 2] >= 0) & (nbr[:, 2] < shape_arr[2])
        )
        if not inb.any():
            continue
        in_idx = np.where(inb)[0]
        nbr_in = nbr[in_idx]
        nbr_step = step_arr[nbr_in[:, 0], nbr_in[:, 1], nbr_in[:, 2]]
        is_older = (nbr_step >= 0) & (nbr_step < next_step)
        if not is_older.any():
            continue
        contrib = in_idx[is_older]
        sum_dir[contrib] += d.astype(np.float32)
        count[contrib] += 1
    return count, sum_dir


def compute_growth(
    vg: VoxelGrid,
    connectivity: Literal[26] = 26,
) -> GrowthResult:
    """26-connectivity BFS from bed seeds. See module docstring for semantics."""
    if connectivity != 26:
        # 6 path still works numerically (always gives length-1 face vectors),
        # but it makes the magnitude info trivial — gate behind explicit ask.
        raise ValueError("only connectivity=26 is supported in this build")

    matrix = vg.matrix
    if matrix.dtype != np.bool_:
        matrix = matrix.astype(bool)

    shape = matrix.shape
    shape_arr = np.array(shape, dtype=np.int32)

    step = np.full(shape, -1, dtype=np.int32)
    vectors = np.zeros(shape + (3,), dtype=np.float32)

    seed = _seed_bed_mask(matrix)
    step[seed] = 0

    sqrt2 = float(np.sqrt(2.0))
    sqrt3 = float(np.sqrt(3.0))

    current = 0
    while True:
        painted = step >= 0
        new_mask = _dilate26(painted) & matrix & ~painted
        if not new_mask.any():
            break

        next_step = current + 1
        new_idx = np.argwhere(new_mask).astype(np.int32)
        n = len(new_idx)

        # Gather older-neighbour contributions, one distance class at a time.
        face_count, face_sum = _accumulate_offsets(new_idx, _OFFSETS_FACE, step, next_step, shape_arr)
        edge_count, edge_sum = _accumulate_offsets(new_idx, _OFFSETS_EDGE, step, next_step, shape_arr)
        vert_count, vert_sum = _accumulate_offsets(new_idx, _OFFSETS_VERTEX, step, next_step, shape_arr)

        # Pick the smallest non-empty distance class per voxel.
        has_face = face_count > 0
        has_edge = (edge_count > 0) & ~has_face
        has_vert = (vert_count > 0) & ~has_face & ~has_edge
        # Sanity: with 26-conn BFS, at least one class must have a contributor.
        # If none does, leave vector at zero (will spot anomalies visually).

        vec = np.zeros((n, 3), dtype=np.float32)

        for mask, sum_dir, target_len in (
            (has_face, face_sum, 1.0),
            (has_edge, edge_sum, sqrt2),
            (has_vert, vert_sum, sqrt3),
        ):
            if not mask.any():
                continue
            d = sum_dir[mask]
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            # Norms can't be zero here (mask implies count > 0, and offset
            # vectors in each class are non-zero). Defensive guard anyway.
            norms[norms == 0] = 1.0
            unit = d / norms
            vec[mask] = unit * target_len

        # Negate: stored direction is OLDER -> NEWER (the growth direction).
        # Magnitudes stay in {1, √2, √3}.
        vec = -vec

        vectors[new_idx[:, 0], new_idx[:, 1], new_idx[:, 2]] = vec
        step[new_mask] = next_step

        current = next_step

    n_steps = int(step.max()) + 1 if (step >= 0).any() else 0
    return GrowthResult(
        step=step,
        vectors=vectors,
        n_steps=n_steps,
        pitch=vg.pitch,
        origin=vg.origin.astype(np.float64).copy(),
        connectivity=connectivity,
    )
