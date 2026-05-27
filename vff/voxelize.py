from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class VoxelGrid:
    """Solid voxelization result aligned to world coordinates.

    matrix[i, j, k] is True iff voxel (i, j, k) is inside the mesh.
    The center of voxel (i, j, k) is at: origin + (i + 0.5, j + 0.5, k + 0.5) * pitch
    (i.e. `origin` is the corner of voxel (0, 0, 0), not its center).
    """

    matrix: np.ndarray  # shape (nx, ny, nz), dtype bool
    origin: np.ndarray  # shape (3,), world position of the corner of voxel (0,0,0)
    pitch: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.matrix.shape  # type: ignore[return-value]

    @property
    def filled_count(self) -> int:
        return int(self.matrix.sum())

    @property
    def world_bounds(self) -> tuple[float, float, float, float, float, float]:
        nx, ny, nz = self.matrix.shape
        ox, oy, oz = self.origin
        return (
            float(ox), float(ox + nx * self.pitch),
            float(oy), float(oy + ny * self.pitch),
            float(oz), float(oz + nz * self.pitch),
        )


def voxelize_solid(mesh: trimesh.Trimesh, pitch: float, pad: int = 1) -> VoxelGrid:
    """Solid voxelization via point-in-mesh test on grid cell centers.

    Uses trimesh's mesh.contains() which (with embreex installed) does fast
    ray-based inside/outside testing. Exact for watertight meshes.

    `pad` adds N voxels of empty padding around the mesh bounding box so the
    rendered grid breathes a bit and rounding never clips a surface voxel.
    """
    lo = mesh.bounds[0] - pad * pitch
    hi = mesh.bounds[1] + pad * pitch
    n = np.ceil((hi - lo) / pitch).astype(int)
    n = np.maximum(n, 1)

    # Snap the origin to a clean multiple of pitch so the same grid lines up
    # across re-voxelizations at the same pitch — visually stable.
    origin = np.floor(lo / pitch) * pitch

    cx = origin[0] + (np.arange(n[0]) + 0.5) * pitch
    cy = origin[1] + (np.arange(n[1]) + 0.5) * pitch
    cz = origin[2] + (np.arange(n[2]) + 0.5) * pitch
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    inside = mesh.contains(points)
    matrix = inside.reshape(n[0], n[1], n[2])

    return VoxelGrid(matrix=matrix, origin=origin, pitch=float(pitch))
