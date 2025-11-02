import sys
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple
from utils.Decorator import time_counter
import os

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
) -> Tuple[NDArray[np.floating[Any]], NDArray[np.uint32]]:
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
        Tuple[NDArray[np.floating[Any]], NDArray[np.uint32]]: Vertices and Indices
    """

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


def perspective_projection(
    left: np.float32,
    right: np.float32,
    bottom: np.float32,
    top: np.float32,
    near: np.float32,
    far: np.float32,
) -> NDArray[np.floating[Any]]:
    """
    Calculate the perspective projection matrix
    Replace the method glFrustum in order to use the core mode of the shaders

    Args:
        left (np.float32)
        right (np.float32)
        bottom (np.float32)
        top (np.float32)
        near (np.float32)
        far (np.float32)

    Returns:
        NDArray[np.floating[Any]]
    """
    Matrix = np.zeros((4, 4), dtype=np.float32)
    Matrix[0, 0] = 2 * near / (right - left)
    Matrix[0, 2] = (right + left) / (right - left)
    Matrix[1, 1] = 2 * near / (top - bottom)
    Matrix[1, 2] = (top + bottom) / (top - bottom)
    Matrix[2, 2] = -(far + near) / (far - near)
    Matrix[2, 3] = -2 * far * near / (far - near)
    Matrix[3, 2] = -1
    return Matrix


def ortho_projection(
    left: np.float32,
    right: np.float32,
    bottom: np.float32,
    top: np.float32,
    near: np.float32,
    far: np.float32,
) -> NDArray[np.floating[Any]]:
    """
    Calculate the orthogonal projection matrix
    Replace the method glOrtho in order to use the core mode of the shaders

    Args:
        left (np.float32)
        right (np.float32)
        bottom (np.float32)
        top (np.float32)
        near (np.float32)
        far (np.float32)

    Returns:
        NDArray[np.floating[Any]]
    """
    Matrix = np.zeros((4, 4), dtype=np.float32)
    Matrix[0, 0] = 2 / (right - left)
    Matrix[0, 3] = -(right + left) / (right - left)
    Matrix[1, 1] = 2 / (top - bottom)
    Matrix[1, 3] = -(top + bottom) / (top - bottom)
    Matrix[2, 2] = -2 / (far - near)
    Matrix[2, 3] = -(far + near) / (far - near)
    Matrix[3, 3] = 1
    return Matrix


def lookat(
    EYE: NDArray[np.floating[Any]],
    LOOK_AT: NDArray[np.floating[Any]],
    EYE_UP: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """
    Calculate the View matrix
    Replace the method gluLookAt in order to use the core mode of the shaders

    Args:
        EYE (NDArray[np.floating[Any]])
        LOOK_AT (NDArray[np.floating[Any]])
        EYE_UP (NDArray[np.floating[Any]])

    Returns:
        NDArray[np.floating[Any]]
    """
    CameraDirection: NDArray[np.floating[Any]] = LOOK_AT - EYE
    CameraDirection = CameraDirection / np.linalg.norm(CameraDirection)

    CameraRight: NDArray[np.floating[Any]] = np.cross(CameraDirection, EYE_UP)
    CameraRight = CameraRight / np.linalg.norm(CameraRight)

    CameraUP: NDArray[np.floating[Any]] = np.cross(CameraRight, CameraDirection)
    Translation: NDArray[np.floating[Any]] = np.array(
        [
            -np.dot(CameraRight, EYE),
            -np.dot(CameraUP, EYE),
            np.dot(CameraDirection, EYE),
        ],
        dtype=np.float32,
    )
    Empty: NDArray[np.floating[Any]] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    ViewMatrix: NDArray[np.floating[Any]] = np.column_stack(
        (CameraRight, CameraUP, -CameraDirection, Translation)
    )
    ViewMatrix = np.vstack((ViewMatrix, Empty))

    return ViewMatrix


def scalef(scale: NDArray[np.floating[Any]]):
    """
    Calculate the scaling matrix
    Replace the method glScalef in order to use the core mode of the shaders

    Args:
        scale (float)
    """

    return np.array(
        [[scale[0], 0, 0, 0], [0, scale[1], 0, 0], [0, 0, scale[2], 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )

def load_shader(file_path: str):
        """_summary_
        Load shader from file_path

        Args:
            file_path (str): the file path

        Returns:
            the shader
        """
        if getattr(sys, "frozen", False):
            BasePth = sys._MEIPASS  # type: ignore[attr-defined]
        else:
            BasePth = os.path.abspath(".")
        dir = os.path.join(BasePth, file_path) # type: ignore
        with open(dir, "r") as f:
            return f.read()
        
