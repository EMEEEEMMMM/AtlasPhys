import numpy as np
from numpy.typing import NDArray
from numba import njit  # type: ignore

GRAVITY: NDArray[np.float32] = np.array([0.0, -9.8, 0.0], dtype=np.float32)


def rotate_vector(
    initial_vector: NDArray[np.float32],
    axis: NDArray[np.float32],
    angle: float,
) -> NDArray[np.float32]:
    """_summary_
    Based on the Rodrigues' rotation formula, rotate the vector v around the axis by angle

    Args:
        initial_vector (NDArray[np.float32]): The initial vector which will be rotate of angle
        axis (NDArray[np.float32]): The rotation axis
        angle (float): The rotation angle

    Returns:
        NDArray[np.float32]: The vector that has been rotated
    """

    Normal_Axis: np.float32 = np.linalg.norm(axis)
    # Vector K
    Normalized_Axis: NDArray[np.float32] = axis / Normal_Axis

    # Calculate the cos and sin of angle
    angle_cos: float = np.cos(angle)
    angle_sin: float = np.sin(angle)

    # Vector k X Vector V
    VK_CrossProduct: NDArray[np.float32] = np.cross(Normalized_Axis, initial_vector)

    # The projection length of Vector v on Vector k * Length of Vector k
    VK_DotProduct: NDArray[np.float32] = np.dot(Normalized_Axis, initial_vector)

    # Vector V ratated
    V_Rotated: NDArray[np.float32] = (
        initial_vector * angle_cos
        + (1 - angle_cos) * VK_DotProduct * Normalized_Axis
        + VK_CrossProduct * angle_sin
    ).astype(np.float32)

    return V_Rotated


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def normalize(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    To normalize a array

    Args:
        arr (Ndarray): The array need to normalize

    Returns:
        Normalized array
    """
    Normalized: np.float32 = np.linalg.norm(arr)

    if Normalized == 0:
        return np.zeros_like(arr)

    return arr / Normalized


def scale_matrix(Sx: float, Sy: float, Sz: float) -> NDArray[np.float32]:
    """
    To create a scale matrix in order to scale the target matrix

    Args:
        Sx (float): scaling factor of x
        Sy (float): scaling factor of y
        Sz (float): scaling factor of z

    Returns:
        NDArray[np.float32]: a 4x4 matrix
    """
    ScaleMatrix: NDArray[np.float32] = np.identity(4, dtype=np.float32)
    Rows, Cols = np.diag_indices(4)
    ScaleMatrix[Rows, Cols] = [Sx, Sy, Sz, 1.0]

    return ScaleMatrix


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def translation_matrix(Tx: float, Ty: float, Tz: float) -> NDArray[np.float32]:
    """
    To create a translation matrix in order to translate a matrix

    Args:
        Tx (float): the amount of translation on x axis
        Ty (float): the amount of translation on y axis
        Tz (float): the amount of translation on z axis

    Returns:
        NDArray[np.float32]: a 4x4 matrix
    """
    TranslationMatrix: NDArray[np.float32] = np.identity(4, dtype=np.float32)
    TranslationMatrix[0, 3] = Tx
    TranslationMatrix[1, 3] = Ty
    TranslationMatrix[2, 3] = Tz

    return TranslationMatrix


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def _rotation_matrix_x(angle: float) -> NDArray[np.float32]:
    """
    To create a rotation matrix in order to rotate the target matrix around the x axis

    Args:
        angle (float): angle to rotate (not in radians)

    Returns:
        NDArray[np.float32]: a 4x4 matrix
    """
    angle = np.radians(angle)
    RotationMatrix: NDArray[np.float32] = np.identity(4, dtype=np.float32)
    RotationMatrix[1, 1] = np.cos(angle)
    RotationMatrix[1, 2] = -np.sin(angle)
    RotationMatrix[2, 1] = np.sin(angle)
    RotationMatrix[2, 2] = np.cos(angle)

    return RotationMatrix


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def _rotation_matrix_y(angle: float) -> NDArray[np.float32]:
    """
    To create a rotation matrix in order to rotate the target matrix around the y axis

    Args:
        angle (float): angle to rotate (not in radians)

    Returns:
        NDArray[np.float32]: a 4x4 matrix
    """
    angle = np.radians(angle)
    RotationMatrix: NDArray[np.float32] = np.identity(4, dtype=np.float32)
    RotationMatrix[0, 0] = np.cos(angle)
    RotationMatrix[0, 2] = np.sin(angle)
    RotationMatrix[2, 0] = -np.sin(angle)
    RotationMatrix[2, 2] = np.cos(angle)

    return RotationMatrix


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def _rotation_matrix_z(angle: float) -> NDArray[np.float32]:
    """
    To create a rotation matrix in order to rotate the target matrix around the z axis

    Args:
        angle (float): angle to rotate (not in radians)

    Returns:
        NDArray[np.float32]: a 4x4 matrix
    """
    angle = np.radians(angle)
    RotationMatrix: NDArray[np.float32] = np.identity(4, dtype=np.float32)
    RotationMatrix[0, 0] = np.cos(angle)
    RotationMatrix[0, 1] = -np.sin(angle)
    RotationMatrix[1, 0] = np.sin(angle)
    RotationMatrix[1, 1] = np.cos(angle)

    return RotationMatrix


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def rotation_matrix(
    Angle_X: float, Angle_Y: float, Angle_Z: float
) -> NDArray[np.float32]:
    """
    To generate a rotation matrix for every P_Objects

    Args:
        Angle_X (float)
        Angle_Y (float)
        Angle_Z (float)

    Returns:
        NDArray[np.float32]
    """

    Rot_Matrix_X: NDArray[np.float32] = _rotation_matrix_x(Angle_X)
    Rot_Matrix_Y: NDArray[np.float32] = _rotation_matrix_y(Angle_Y)
    Rot_Matrix_Z: NDArray[np.float32] = _rotation_matrix_z(Angle_Z)

    Rot_Matrix: NDArray[np.float32] = Rot_Matrix_Z @ Rot_Matrix_Y @ Rot_Matrix_X  # type: ignore

    return Rot_Matrix


def euler_to_matrix(e: NDArray[np.float32]) -> NDArray[np.float32]:
    cx, cy, cz = np.cos(e)
    sx, sy, sz = np.sin(e)

    Rx: NDArray[np.float32] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0, cx, -sx],
            [0, sx, cx],
        ],
        dtype=np.float32,
    )

    Ry: NDArray[np.float32] = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0, cy],
        ],
        dtype=np.float32,
    )

    Rz: NDArray[np.float32] = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return Rz @ Ry @ Rx  # type: ignore
