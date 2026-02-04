import numpy as np
from numpy.typing import NDArray
from typing import Callable
from numba import njit  # type: ignore

from utils import SimplexSolver
from utils.G_Object import P_Object
from utils.GJK import check_collison, _support_function_polygon  # type: ignore
from utils.EPA import epa
from utils.MathPhys_utils import GRAVITY, euler_to_matrix

PENETRATION_SLOP: float = 0.01
CORRECTION_PERCENTAGE: float = 0.2


# @njit(nogil=True, fastmath=True, cache=True, parallel=False)  # type: ignore
def integrator(
    DynamicObjects: list[P_Object],
    DeltaTime: float,
) -> None:
    """
    The integrator which applies gravity to the object and update its position based on the velocity and deltatime
    Update: numba version

    Args:
        Objects (list[P_Object]): The object that need to be updated
        Deltatime (float)
    """

    for obj in DynamicObjects:
        obj.Velocity += GRAVITY * DeltaTime
        obj.Position += obj.Velocity * DeltaTime

        Tau: NDArray[np.float32] = np.zeros(3, dtype=np.float32)
        obj.AngularVelocity += (obj.InvInertiaWorld @ Tau) * DeltaTime
        obj.Rotation += obj.AngularVelocity * DeltaTime
        obj.RotationMatrix = euler_to_matrix(obj.Rotation)

        obj.InvInertiaWorld = (
            obj.RotationMatrix @ obj.InvInertiaBody @ np.transpose(obj.RotationMatrix)
        ).astype(np.float32)


def the_collision(
    ObjectA: P_Object, ObjectB: P_Object
) -> tuple[bool, NDArray[np.float32], float]:
    """
    Determine whether the collision had happened
    If happened, then move the two object based the normal and the depth of the Penetration

    Args:
        ObjectA (P_Object)
        ObjectB (P_Object)
    """
    IsColliding: bool
    Simplex: list[NDArray[np.float32]]
    IsColliding, Simplex = check_collison(ObjectA, ObjectB)

    Support_fn: Callable[[NDArray[np.float32]], NDArray[np.float32]] = (
        SimplexSolver.make_support_fn(ObjectA, ObjectB)
    )

    Normal: NDArray[np.float32] = np.array([], dtype=np.float32)
    Depth: float = 0.0

    if len(Simplex) == 4:
        Normal, Depth = epa(Simplex, ObjectA, ObjectB)

    elif len(Simplex) == 3:
        Tetra: list[NDArray[np.float32]] | None = SimplexSolver.expand_triangle_to_tetra(Simplex, Support_fn)  # type: ignore
        if Tetra:
            Normal, Depth = epa(Tetra, ObjectA, ObjectB)

        else:
            Normal, Depth = SimplexSolver.triangle_contact(Simplex)

    elif len(Simplex) == 2:
        Tetra: list[NDArray[np.float32]] = SimplexSolver.expand_segment_to_tetra(
            Simplex, Support_fn
        )
        Normal, Depth = epa(Tetra, ObjectA, ObjectB)

    return IsColliding, Normal, Depth  # type: ignore
