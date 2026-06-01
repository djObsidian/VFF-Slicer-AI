from .build_volume import BuildVolume
from .mesh_io import load_and_place
from .voxelize import voxelize_solid, VoxelGrid
from .growth import compute_growth, clamp_to_vertical, GrowthResult
from .deform import deform_mesh

__all__ = [
    "BuildVolume",
    "load_and_place",
    "voxelize_solid",
    "VoxelGrid",
    "compute_growth",
    "clamp_to_vertical",
    "GrowthResult",
    "deform_mesh",
    "Viewer",
]


def __getattr__(name: str):
    # Lazy: the viewer pulls in pyvista/VTK, which the headless paths
    # (--no-viewer, --gcode-in, batch export) don't need. Importing the
    # package must not require a GUI stack — only touch it on demand.
    if name == "Viewer":
        from .viewer import Viewer
        return Viewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
