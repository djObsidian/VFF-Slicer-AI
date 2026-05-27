import numpy as np
import trimesh

from .build_volume import BuildVolume


def load_and_place(stl_path: str, volume: BuildVolume) -> trimesh.Trimesh:
    """Load an STL and place it on the build plate.

    Convention: Z is up. The mesh is translated so its minimum Z sits at 0
    (on the bed) and its X/Y bounding box is centered on the build volume.
    """
    mesh = trimesh.load(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a single mesh: {type(mesh).__name__}")

    z_min = mesh.bounds[0, 2]
    mesh.apply_translation([0.0, 0.0, -z_min])

    bounds = mesh.bounds
    cx = 0.5 * (bounds[0, 0] + bounds[1, 0])
    cy = 0.5 * (bounds[0, 1] + bounds[1, 1])
    tx, ty = volume.center_xy
    mesh.apply_translation([tx - cx, ty - cy, 0.0])

    return mesh
