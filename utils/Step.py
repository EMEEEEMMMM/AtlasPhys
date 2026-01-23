import numpy as np
from numpy.typing import NDArray
import os

from utils.G_Object import P_Object
from utils.GJK import check_collison
from utils.EPA import epa
from utils.MathPhys_utils import linear_scalar_impulse

MAX_WORKERS: int = os.cpu_count()  # type: ignore


def integrator(Objects: list[P_Object], Deltatime: float) -> None:
    """
    The integrator which applies gravity to the object and update its position based on the velocity and deltatime

    Args:
        Objects (list[P_Object]): The object that need to be updated
        Deltatime (float)
    """
    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     executor.map(lambda obj: _update_obj(obj, Deltatime), Objects) # type: ignore
    for obj in Objects:
        if not obj.Collidable:
            continue

        if obj.ReciprocalMass == 0.0:
            continue

        obj.update_position(Deltatime)


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
        e = 0.0

    Scalar_Impulse: float = linear_scalar_impulse(e, VrelNormal, InvMassA, InvMassB)

    Impulse: NDArray[np.float32] = Scalar_Impulse * Normal

    ObjectA.Velocity += InvMassA * Impulse
    ObjectB.Velocity -= InvMassB * Impulse
