"""Spatial growth algorithm: BFS from the printer-bed surface upward through
a solid-voxelized model. Produces

  - per-voxel step index (0 = bed seed, 1..N = BFS layers, -1 = outside model)
  - per-voxel unit vector pointing from each newly-painted voxel toward the
    average of its step-(n-1) face neighbours (i.e. "toward the source / bed")

This is what the user wants to feed into a non-planar slicer: it encodes how
the part "grows" from the build plate layer by physical-contact layer, and
the vector field at each step tells the slicer the local "down toward source"
direction.

For face-connectivity (6) every newly-painted voxel always has at least one
face neighbour at step n-1 (BFS invariant), so the vector field is always
well-defined for n >= 1. Step-0 voxels (bed seeds) carry a zero vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .voxelize import VoxelGrid


# Face-neighbour offsets (6-connectivity).
_OFFSETS_6 = np.array(
    [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ],
    dtype=np.int32,
)

# Full 26-neighbour offsets (face + edge + vertex).
_OFFSETS_26 = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1)
     for dy in (-1, 0, 1)
     for dz in (-1, 0, 1)
     if (dx, dy, dz) != (0, 0, 0)],
    dtype=np.int32,
)


@dataclass
class GrowthResult:
    step: np.ndarray       # int32 (nx, ny, nz). -1 = outside model; 0 = bed seed; 1..n_steps-1 = BFS layer.
    vectors: np.ndarray    # float32 (nx, ny, nz, 3). Unit "toward older" vector per painted voxel; zero for step 0 and outside.
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
    """Voxels in the lowest non-empty Z layer of the model become seeds.

    We don't pin to a specific world-Z value: as long as the mesh sits on the
    bed (Z_min=0, enforced by load_and_place), this is the first layer above
    the plate. Using "lowest non-empty layer" instead also handles the case
    where the voxel grid is padded below the bed by one row of empty voxels.
    """
    nz = matrix.shape[2]
    seed = np.zeros_like(matrix)
    for k in range(nz):
        if matrix[:, :, k].any():
            seed[:, :, k] = matrix[:, :, k]
            return seed
    return seed  # empty model


def _dilate6(mask: np.ndarray) -> np.ndarray:
    """6-connectivity binary dilation, pure numpy. Faster than scipy here
    because we don't carry the dependency and only need one structure."""
    out = mask.copy()
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def _dilate26(mask: np.ndarray) -> np.ndarray:
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


def compute_growth(
    vg: VoxelGrid,
    connectivity: Literal[6, 26] = 6,
) -> GrowthResult:
    """BFS from bed seeds outward through the model.

    Each iteration:
      1. Dilate the painted mask by one voxel within the model.
      2. The newly-painted voxels get step = current+1.
      3. For each new voxel, scan its face neighbours, collect those at
         step current (the previous frontier), and accumulate their offset
         vectors. The final per-voxel vector is the normalized sum (i.e.,
         unit direction toward the centroid of contributing older neighbours).

    For 6-connectivity, the BFS invariant guarantees at least one face
    neighbour at step (current) for every new voxel. For 26-connectivity a
    voxel may be reached via an edge/vertex neighbour only; we still look at
    face neighbours first (they are the nearest by Euclidean distance), and
    fall back to all 26 if no face neighbour is at the previous step.
    """
    if connectivity not in (6, 26):
        raise ValueError(f"connectivity must be 6 or 26, got {connectivity}")

    matrix = vg.matrix
    if matrix.dtype != np.bool_:
        matrix = matrix.astype(bool)

    shape = matrix.shape
    shape_arr = np.array(shape, dtype=np.int32)

    step = np.full(shape, -1, dtype=np.int32)
    vectors = np.zeros(shape + (3,), dtype=np.float32)

    seed = _seed_bed_mask(matrix)
    step[seed] = 0

    dilate = _dilate6 if connectivity == 6 else _dilate26
    # Vector lookup always uses the same neighbourhood as the dilation.
    offsets = _OFFSETS_6 if connectivity == 6 else _OFFSETS_26

    current = 0
    while True:
        painted = step >= 0
        new_mask = dilate(painted) & matrix & ~painted
        if not new_mask.any():
            break

        next_step = current + 1
        new_idx = np.argwhere(new_mask).astype(np.int32)  # (N, 3)
        n_new = len(new_idx)
        vec_acc = np.zeros((n_new, 3), dtype=np.float32)

        for d in offsets:
            nbr = new_idx + d  # (N, 3)
            inb = (
                (nbr[:, 0] >= 0) & (nbr[:, 0] < shape_arr[0])
                & (nbr[:, 1] >= 0) & (nbr[:, 1] < shape_arr[1])
                & (nbr[:, 2] >= 0) & (nbr[:, 2] < shape_arr[2])
            )
            if not inb.any():
                continue
            in_indices = np.where(inb)[0]
            nbr_in = nbr[in_indices]
            nbr_step = step[nbr_in[:, 0], nbr_in[:, 1], nbr_in[:, 2]]
            from_prev = nbr_step == current  # previous frontier (= step before next_step)
            if not from_prev.any():
                continue
            contrib_idx = in_indices[from_prev]
            vec_acc[contrib_idx] += d.astype(np.float32)

        norms = np.linalg.norm(vec_acc, axis=1, keepdims=True)
        zero_mask = (norms[:, 0] == 0)
        # Should not happen for connectivity=6 (BFS invariant); leave zero
        # vectors as-is so the caller can spot the anomaly.
        norms[zero_mask, 0] = 1.0
        unit = vec_acc / norms

        vectors[new_idx[:, 0], new_idx[:, 1], new_idx[:, 2]] = unit
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
