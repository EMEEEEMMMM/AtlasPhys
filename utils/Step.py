import numpy as np
from numpy.typing import NDArray
import os
from numba import njit  # type: ignore

from utils.G_Object import P_Object
from utils.GJK import check_collison
from utils.EPA import epa
from utils.MathPhys_utils import linear_scalar_impulse, GRAVITY

MAX_WORKERS: int = os.cpu_count()  # type: ignore


@njit(nogil=True, fastmath=True, cache=True, parallel=False)    # type: ignore
def integrator(
    Positions: NDArray[np.float32], Velocities: NDArray[np.float32], DeltaTime: float
) -> None:
    """
    The integrator which applies gravity to the object and update its position based on the velocity and deltatime
    Update: numba version

    Args:
        Objects (list[P_Object]): The object that need to be updated
        Deltatime (float)
    """
    N: int = len(Positions)

    for i in range(N):
        Velocities[i] += GRAVITY * DeltaTime
        Positions[i] += Velocities[i] * DeltaTime


def extract_data(
    Objects: list[P_Object],
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[P_Object]]:
    """
    Extract positions and velocities of every P_Object(except the Plane and Coordinate_axis)
    To support for the numba version of the integrator

    Args:
        Objects (list[P_Object])

    Returns:
        NDArray[np.float32]: positions, velocities, dynamic objects
    """
    DynamicObjects: list[P_Object] = []
    for obj in Objects:
        if obj.Shape not in ["Plane", "CoordinateAxis"] and obj.Collidable:
            DynamicObjects.append(obj)

    N: int = len(DynamicObjects)
    Positions: NDArray[np.float32] = np.zeros((N, 3), dtype=np.float32)
    Velocities: NDArray[np.float32] = np.zeros((N, 3), dtype=np.float32)

    for i, obj in enumerate(DynamicObjects):
        Positions[i] = obj.Position
        Velocities[i] = obj.Velocity

    return Positions, Velocities, DynamicObjects

def update_data(NewPositions: NDArray[np.float32], NewVelocities: NDArray[np.float32], DynamicObjects: list[P_Object]) -> None:
    """
    To update the data from the integrator to the objects

    Args:
        NewPositions (NDArray[np.float32])
        NewVelocities (NDArray[np.float32])
        DynamicObjects (list[P_Object])
    """
    for i, obj in enumerate(DynamicObjects):
        obj.Position = NewPositions[i]
        obj.Velocity = NewVelocities[i]


def the_collision(ObjectA: P_Object, ObjectB: P_Object) -> None:
    """
    Determine whether the collision had happened
    If happened, then move the two object based the normal and the depth of the penetration

    Args:
        ObjectA (P_Object)
        ObjectB (P_Object)
    """
    if not ObjectA.Collidable or not ObjectB.Collidable:
        return

    IsColliding: bool
    Simplex: list[NDArray[np.float32]]
    IsColliding, Simplex = check_collison(ObjectA, ObjectB)

    if not IsColliding:
        return

    Normal: NDArray[np.float32]
    Depth: float
    Normal, Depth = epa(Simplex, ObjectA, ObjectB)

    InvMassA: float = ObjectA.ReciprocalMass
    InvMassB: float = ObjectB.ReciprocalMass
    Percentage: float = 0.8
    Slope: float = 0.001
    TotalMass_Reciprocal: float = ObjectA.ReciprocalMass + ObjectB.ReciprocalMass
    CorrectionDepth: float = max(Depth - Slope, 0.0)

    if TotalMass_Reciprocal > 0.0:
        CorrectionA = (
            CorrectionDepth * InvMassA / TotalMass_Reciprocal * Percentage * Normal
        )
        CorrectionB = (
            CorrectionDepth * InvMassB / TotalMass_Reciprocal * Percentage * Normal
        )
        ObjectA.Position -= CorrectionA
        ObjectB.Position += CorrectionB

    RelativeVelocity: NDArray[np.float32] = ObjectA.Velocity - ObjectB.Velocity
    VrelNormal: float = np.dot(
        RelativeVelocity, Normal
    )  # relative velocity along the normal

    if VrelNormal < 0:
        return

    e: float = min(ObjectA.Restitution, ObjectB.Restitution)

    if VrelNormal < 0.5:
        e *= 0.5

    Scalar_Impulse: float = linear_scalar_impulse(e, VrelNormal, InvMassA, InvMassB)

    Impulse: NDArray[np.float32] = Scalar_Impulse * Normal

    ObjectA.Velocity += InvMassA * Impulse
    ObjectB.Velocity -= InvMassB * Impulse
