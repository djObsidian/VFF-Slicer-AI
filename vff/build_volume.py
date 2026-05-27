from dataclasses import dataclass
import numpy as np


@dataclass
class BuildVolume:
    size: np.ndarray  # (3,) mm — extents along X, Y, Z. Origin at (0, 0, 0), Z up.

    @classmethod
    def cube(cls, side: float = 250.0) -> "BuildVolume":
        return cls(size=np.array([side, side, side], dtype=np.float64))

    @classmethod
    def of(cls, x: float, y: float, z: float) -> "BuildVolume":
        return cls(size=np.array([x, y, z], dtype=np.float64))

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        sx, sy, sz = self.size
        return (0.0, sx, 0.0, sy, 0.0, sz)

    @property
    def center(self) -> np.ndarray:
        return self.size / 2.0

    @property
    def center_xy(self) -> np.ndarray:
        return self.size[:2] / 2.0
