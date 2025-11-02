import numpy as np
from numpy.typing import NDArray
from typing import Any

GRAVITY: float = 9.8



def rotate_vector(
    initial_vector: NDArray[np.floating[Any]],
    axis: NDArray[np.floating[Any]],
    angle: float,
) -> NDArray[np.floating[Any]]:
    """_summary_
    Based on the Rodrigues' rotation formula, rotate the vector v around the axis by angle

    Args:
        initial_vector (NDArray[np.floating[Any]]): The initial vector which will be rotate of angle
        axis (NDArray[np.floating[Any]]): The rotation axis
        angle (float): The rotation angle

    Returns:
        NDArray[np.floating[Any]]: The vector that has been rotated
    """

    Normal_Axis: np.floating[Any] = np.linalg.norm(axis)
    # Vector K
    Normalized_Axis: NDArray[np.floating[Any]] = axis / Normal_Axis

    # Calculate the cos and sin of angle
    angle_cos: float = np.cos(angle)
    angle_sin: float = np.sin(angle)

    # Vector k X Vector V
    VK_CrossProduct: NDArray[np.floating[Any]] = np.cross(
        Normalized_Axis, initial_vector
    )

    # The projection length of Vector v on Vector k * Length of Vector k
    VK_DotProduct: NDArray[np.floating[Any]] = np.dot(Normalized_Axis, initial_vector)

    # Vector V ratated
    V_Rotated: NDArray[np.floating[Any]] = (
        initial_vector * angle_cos
        + (1 - angle_cos) * VK_DotProduct * Normalized_Axis
        + VK_CrossProduct * angle_sin
    )

    return V_Rotated

def normalize(arr: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        To normalize a array

        Args:
            arr (Ndarray): The array need to normalize 

        Returns:
            Normalized array
        """
        Normalized: np.floating[Any] = np.linalg.norm(arr)
        
        return arr / Normalized