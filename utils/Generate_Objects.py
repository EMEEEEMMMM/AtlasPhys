from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from utils.Decorator import time_counter
from OpenGL.GL import * # type: ignore

@time_counter
def add_triangle(
    Side_Length: float,
    X_Coordinate: float,
    Y_Coordinate: float,
    Z_Coordinate: float,
    R_v: float,
    G_v: float,
    B_v: float,
    A_v: float,
) -> dict[str, int | NDArray[np.float32] | NDArray[np.uint32]]:
    # fmt:off
    X, Y, Z = 0.0, 0.0, 0.0
    Vertices: NDArray[np.float32] = np.array([
        X - (Side_Length / 2), Y - (np.power(3,1 / 2) / 6 * Side_Length), Z, R_v, G_v, B_v, A_v,
        X + (Side_Length / 2), Y - (np.power(3,1 / 2) / 6 * Side_Length), Z, R_v, G_v, B_v, A_v,
        X, Y + (np.power(3,1 / 2) / 3 * Side_Length), Z, R_v, G_v, B_v, A_v,
        ],dtype=np.float32,
    )
    Indices: NDArray[np.uint32] = np.array([0, 1, 2], dtype=np.uint32)
    # fmt: on

    ObjectData: dict[str, int | NDArray[np.float32] | NDArray[np.uint32]] = {
        "GL_Type": GL_TRIANGLES,
        "Vertices": Vertices,
        "Indices": Indices,
    }

    return ObjectData


@time_counter
def add_square(
    Side_Length: float,
    X_Coordinate: float,
    Y_Coordinate: float,
    Z_Coordinate: float,
    R_v: float,
    G_v: float,
    B_v: float,
    A_v: float,
) -> dict[str, int | NDArray[np.float32] | NDArray[np.uint32]]:
    X, Y, Z = 0.0, 0.0, 0.0
    #  fmt: off
    Vertices: NDArray[np.float32] = np.array([
        X - (Side_Length / 2), Y + (Side_Length / 2), Z, R_v, G_v, B_v, A_v,
        X + (Side_Length / 2), Y + (Side_Length / 2), Z, R_v, G_v, B_v, A_v,
        X + (Side_Length / 2), Y - (Side_Length / 2), Z, R_v, G_v, B_v, A_v,
        X - (Side_Length / 2), Y - (Side_Length / 2), Z, R_v, G_v, B_v, A_v,
        ], dtype=np.float32
    )
    Indices: NDArray[np.uint32] = np.array([0, 1, 2, 3], dtype=np.uint32)
    # fmt: on

    ObjectData: dict[str, int | NDArray[np.float32] | NDArray[np.uint32]] = {
        "GL_Type": GL_QUADS,
        "Vertices": Vertices,
        "Indices": Indices,
    }

    return ObjectData


@time_counter
def add_cube(
    Side_Length: float,
    X_Coordinate: float,
    Y_Coordinate: float,
    Z_Coordinate: float,
    R_v: float,
    G_v: float,
    B_v: float,
    A_v: float,
) -> dict[str, int | NDArray[np.float32] | NDArray[np.uint32]]:
    #    v4----- v5
    #   /|      /|
    #  v0------v1|
    #  | |     | |
    #  | v7----|-v6
    #  |/      |/
    #  v3------v2
    X, Y, Z = 0.0, 0.0, 0.0

    # fmt: off
    Vertices: NDArray[np.float32] = np.array([
        X - (Side_Length / 2), Y + (Side_Length / 2), Z + (Side_Length / 2), R_v, G_v, B_v, A_v,   # v0
        X + (Side_Length / 2), Y + (Side_Length / 2), Z + (Side_Length / 2), R_v, G_v, B_v, A_v,   # v1
        X + (Side_Length / 2), Y - (Side_Length / 2), Z + (Side_Length / 2), R_v, G_v, B_v, A_v,   # v2
        X - (Side_Length / 2), Y - (Side_Length / 2), Z + (Side_Length / 2), R_v, G_v, B_v, A_v,   # v3
        X - (Side_Length / 2), Y + (Side_Length / 2), Z - (Side_Length / 2), R_v, G_v, B_v, A_v,   # v4
        X + (Side_Length / 2), Y + (Side_Length / 2), Z - (Side_Length / 2), R_v, G_v, B_v, A_v,   # v5
        X + (Side_Length / 2), Y - (Side_Length / 2), Z - (Side_Length / 2), R_v, G_v, B_v, A_v,   # v6
        X - (Side_Length / 2), Y - (Side_Length / 2), Z - (Side_Length / 2), R_v, G_v, B_v, A_v,   # v7
    ], dtype=np.float32)
    Indices: NDArray[np.uint32] = np.array([
        0, 1, 2, 3, # v0-v1-v2-v3
        4, 5, 1, 0, # v4-v5-v1-v0
        3, 2, 6, 7, # v3-v2-v6-v7
        5, 4, 7, 6, # v5-v4-v7-v6
        1, 5, 6, 2, # v1-v5-v6-v2
        4, 0, 3, 7  # v4-v0-v3-v7
    ], dtype=np.uint32)
    # fmt: on
    ObjectData: dict[str, int | NDArray[np.float32] | NDArray[np.uint32]] = {
        "GL_Type": GL_QUADS,
        "Vertices": Vertices,
        "Indices": Indices,
    }

    return ObjectData


@time_counter
def add_sphere(
    Side_Length: float,
    X_Coordinate: float,
    Y_Coordinate: float,
    Z_Coordinate: float,
    R_v: float,
    G_v: float,
    B_v: float,
    A_v: float,
    Rings: int = 200,
    Sectors: int = 200,
) -> dict[str, int | NDArray[np.float32] | NDArray[np.uint32]]:
    Vertices, Indices = generate_sphere(
        Side_Length,
        0.0,
        0.0,
        0.0,
        Rings,
        Sectors,
        R_v,
        G_v,
        B_v,
        A_v,
    )
    ObjectData: dict[str, int | NDArray[np.float32] | NDArray[np.uint32]] = {
        "GL_Type": GL_TRIANGLES,
        "Vertices": Vertices,
        "Indices": Indices,
    }

    return ObjectData


@time_counter
def generate_sphere(
    Radius: float,
    X_Coordinate: float,
    Y_Coordinate: float,
    Z_Coordinate: float,
    Rings: int,
    Sectors: int,
    R_v: float = 1.0,
    G_v: float = 1.0,
    B_v: float = 1.0,
    A_v: float = 1.0,
) -> Tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """
    Generate the Vertices and Indices for a sphere

    Args:
        Radius (float): The radius of the sphere
        X_Coordinate (float): The x coordinate of the center of the sphere
        Y_Coordinate (float): The y coordinate of the center of the sphere
        Z_Coordinate (float): The z coordinate of the center of the sphere
        Rings (int): How many parts the sphere will be split in horizontal direction
        Sectors (int): How many parts the sphere will be split in vertical direction
        R_v (float): The color of the sphere
        G_v (float): The color of the sphere
        B_v (float): The color of the sphere
        A_v (float): The color of the sphere

    Returns:
        Tuple[NDArray[np.float32], NDArray[np.uint32]]: Vertices and Indices
    """

    # The Latitude Angle
    Phi: NDArray[np.float32] = np.linspace(
        -np.pi / 2, np.pi / 2, Rings + 1, dtype=np.float32
    )

    # The Longitude Angle
    Theta: NDArray[np.float32] = np.linspace(
        0, 2 * np.pi, Sectors + 1, dtype=np.float32
    )

    # Get the latitude and longitude grid and flatten them out
    PhiGrid: NDArray[np.float32]
    ThetaGrid: NDArray[np.float32]
    PhiGrid, ThetaGrid = np.meshgrid(Phi, Theta, indexing="ij")

    PhiFlattened: NDArray[np.float32] = PhiGrid.flatten()
    ThetaFlattened: NDArray[np.float32] = ThetaGrid.flatten()

    PhiCos: NDArray[np.float32] = np.cos(PhiFlattened)
    PhiSin: NDArray[np.float32] = np.sin(PhiFlattened)
    ThetaCos: NDArray[np.float32] = np.cos(ThetaFlattened)
    ThetaSin: NDArray[np.float32] = np.sin(ThetaFlattened)

    # Suppose the center of the sphere is the origin
    RelativeX: NDArray[np.float32] = Radius * PhiCos * ThetaCos
    RelativeY: NDArray[np.float32] = Radius * PhiSin
    RelativeZ: NDArray[np.float32] = Radius * PhiCos * ThetaSin

    # The real Coordinates
    RealX: NDArray[np.float32] = RelativeX + X_Coordinate
    RealY: NDArray[np.float32] = RelativeY + Y_Coordinate
    RealZ: NDArray[np.float32] = RelativeZ + Z_Coordinate

    R_Array: NDArray[np.float32] = np.full_like(RealX, R_v, dtype=np.float32)
    G_Array: NDArray[np.float32] = np.full_like(RealX, G_v, dtype=np.float32)
    B_Array: NDArray[np.float32] = np.full_like(RealX, B_v, dtype=np.float32)
    A_Array: NDArray[np.float32] = np.full_like(RealX, A_v, dtype=np.float32)

    Vertices: NDArray[np.float32] = np.column_stack(
        [RealX, RealY, RealZ, R_Array, G_Array, B_Array, A_Array]
    ).astype(np.float32).flatten()

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
