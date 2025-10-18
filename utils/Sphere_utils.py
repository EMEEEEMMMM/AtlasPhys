import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple


def generate_sphere(
    Radius: float,
    X_Coordinate: float,
    Y_Coordinate: float,
    Z_Coordinate: float,
    Rings: int,
    Sectors: int,
    R_v: float,
    G_v: float,
    B_v: float,
    A_v: float,
) -> Tuple[NDArray[np.floating[Any]], NDArray[np.uint32]]:

    # The Latitude Angle
    Phi: NDArray[np.floating[Any]] = np.linspace(
        -np.pi / 2, np.pi / 2, Rings + 1, dtype=np.float32
    )

    # The Longitude Angle
    Theta: NDArray[np.floating[Any]] = np.linspace(
        0, 2 * np.pi, Sectors + 1, dtype=np.float32
    )

    # Get the latitude and longitude grid and flatten them out
    PhiGrid: NDArray[np.floating[Any]]
    ThetaGrid: NDArray[np.floating[Any]]
    PhiGrid, ThetaGrid = np.meshgrid(Phi, Theta, indexing="ij")

    PhiFlattened: NDArray[np.floating[Any]] = PhiGrid.flatten()
    ThetaFlattened: NDArray[np.floating[Any]] = ThetaGrid.flatten()

    PhiCos: NDArray[np.floating[Any]] = np.cos(PhiFlattened)
    PhiSin: NDArray[np.floating[Any]] = np.sin(PhiFlattened)
    ThetaCos: NDArray[np.floating[Any]] = np.cos(ThetaFlattened)
    ThetaSin: NDArray[np.floating[Any]] = np.sin(ThetaFlattened)

    # Suppose the center of the sphere is the origin
    RelativeX: NDArray[np.floating[Any]] = Radius * PhiCos * ThetaCos
    RelativeY: NDArray[np.floating[Any]] = Radius * PhiSin
    RelativeZ: NDArray[np.floating[Any]] = Radius * PhiCos * ThetaSin

    # The real Coordinates
    RealX: NDArray[np.floating[Any]] = RelativeX + X_Coordinate
    RealY: NDArray[np.floating[Any]] = RelativeY + Y_Coordinate
    RealZ: NDArray[np.floating[Any]] = RelativeZ + Z_Coordinate

    R_Array: NDArray[np.floating[Any]] = np.full_like(RealX, R_v, dtype=np.float32)
    G_Array: NDArray[np.floating[Any]] = np.full_like(RealX, G_v, dtype=np.float32)
    B_Array: NDArray[np.floating[Any]] = np.full_like(RealX, B_v, dtype=np.float32)
    A_Array: NDArray[np.floating[Any]] = np.full_like(RealX, A_v, dtype=np.float32)

    Vertices: NDArray[np.floating[Any]] = np.column_stack(
        [RealX, RealY, RealZ, R_Array, G_Array, B_Array, A_Array]
    ).astype(np.float32)

    RingIndices: NDArray[np.uint32] = np.arange(Rings, dtype=np.uint32).repeat(Sectors)
    SectorIndices: NDArray[np.uint32] = np.tile(
        np.arange(Sectors, dtype=np.uint32), Rings
    )

    Index: NDArray[np.uint32] = RingIndices * (Sectors + 1) + SectorIndices
    Index1: NDArray[np.uint32] = (RingIndices + 1) * (Sectors + 1) + SectorIndices
    Index2: NDArray[np.uint32] = (RingIndices + 1) * (Sectors + 1) + (SectorIndices + 1)
    Index3: NDArray[np.uint32] = RingIndices * (Sectors + 1) + (SectorIndices + 1)

    Indices: NDArray[np.uint32] = np.concatenate(
        [
            np.column_stack([Index, Index1, Index2]).flatten(),
            np.column_stack([Index, Index2, Index3]).flatten(),
        ]
    ).astype(np.uint32)

    return Vertices, Indices
