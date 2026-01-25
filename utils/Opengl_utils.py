import sys
import numpy as np
from numpy.typing import NDArray
from numba import njit   # type: ignore
from typing import Any
import os
from OpenGL.GL import *  # type: ignore


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def perspective_projection(
    left: np.float32,
    right: np.float32,
    bottom: np.float32,
    top: np.float32,
    near: np.float32,
    far: np.float32,
) -> NDArray[np.float32]:
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
        NDArray[np.float32]
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


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def ortho_projection(
    left: np.float32,
    right: np.float32,
    bottom: np.float32,
    top: np.float32,
    near: np.float32,
    far: np.float32,
) -> NDArray[np.float32]:
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
        NDArray[np.float32]
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


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def lookat(
    EYE: NDArray[np.float32],
    LOOK_AT: NDArray[np.float32],
    EYE_UP: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Calculate the View matrix
    Replace the method gluLookAt in order to use the core mode of the shaders

    Args:
        EYE (NDArray[np.float32])
        LOOK_AT (NDArray[np.float32])
        EYE_UP (NDArray[np.float32])

    Returns:
        NDArray[np.float32]
    """
    CameraDirection: NDArray[np.float32] = LOOK_AT - EYE
    CameraDirection = (CameraDirection / np.linalg.norm(CameraDirection)).astype(
        np.float32
    )

    CameraRight: NDArray[np.float32] = np.cross(CameraDirection, EYE_UP)
    CameraRight = (CameraRight / np.linalg.norm(CameraRight)).astype(np.float32)

    CameraUP: NDArray[np.float32] = np.cross(CameraRight, CameraDirection)
    Translation: NDArray[np.float32] = np.array(
        [
            -np.dot(CameraRight, EYE),
            -np.dot(CameraUP, EYE),
            np.dot(CameraDirection, EYE),
        ],
        dtype=np.float32,
    )
    ViewMatrix: NDArray[np.float32] = np.identity(4, dtype=np.float32)

    ViewMatrix[0, 0] = CameraRight[0]
    ViewMatrix[0, 1] = CameraRight[1]
    ViewMatrix[0, 2] = CameraRight[2]
    ViewMatrix[0, 3] = Translation[0]

    ViewMatrix[1, 0] = CameraUP[0]
    ViewMatrix[1, 1] = CameraUP[1]
    ViewMatrix[1, 2] = CameraUP[2]
    ViewMatrix[1, 3] = Translation[1]

    ViewMatrix[2, 0] = -CameraDirection[0]
    ViewMatrix[2, 1] = -CameraDirection[1]
    ViewMatrix[2, 2] = -CameraDirection[2]
    ViewMatrix[2, 3] = Translation[2]

    return ViewMatrix

@njit(nogil=True, fastmath=True, cache=True)   # type: ignore
def scalef(scale: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Calculate the scaling matrix
    Replace the method glScalef in order to use the core mode of the shaders

    Args:
        scale (float)
    """

    Matrix: NDArray[np.float32] = np.empty((4, 4), dtype=np.float32)
    Matrix.fill(0.0)
    Matrix[0, 0] = scale[0]
    Matrix[1, 1] = scale[1]
    Matrix[2, 2] = scale[2]
    Matrix[3, 3] = 1.0
    return Matrix


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
    dir = os.path.join(BasePth, file_path)  # type: ignore
    with open(dir, "r") as f:
        return f.read()


def analysis_data(
    window: Any,
    Vertices: NDArray[np.float32],
    Indices: NDArray[np.uint32],
) -> tuple[int, int, int]:
    window.makeCurrent()

    vao: int = glGenVertexArrays(1)  # type: ignore
    glBindVertexArray(vao)

    vbo: int = glGenBuffers(1)  # type: ignore
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, Vertices.nbytes, Vertices, GL_STATIC_DRAW)

    ebo: int = glGenBuffers(1)  # type: ignore
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, Indices.nbytes, Indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    return (vao, vbo, ebo)  # type: ignore
