import os
import numpy as np
from numpy.typing import NDArray
from typing import Callable
from numba import njit  # type: ignore

from utils import SimplexSolver
from utils.G_Object import P_Object
from utils.GJK import check_collison
from utils.EPA import epa
from utils.MathPhys_utils import linear_scalar_impulse, GRAVITY

MAX_WORKERS: int = os.cpu_count()  # type: ignore
PENETRATION_SLOP: float = 0.01
CORRECTION_PERCENTAGE: float = 0.2


@njit(nogil=True, fastmath=True, cache=True, parallel=False)  # type: ignore
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


def update_data(
    NewPositions: NDArray[np.float32],
    NewVelocities: NDArray[np.float32],
    DynamicObjects: list[P_Object],
) -> None:
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


def solve_ground_collision_cube(Obj: P_Object, GroundHeight: float = -1.0) -> None:
    """
    Take care of the special case for the "Ground(Plane)" with cubes

    Args:
        Object (P_Object): cube
        GroundHeight (float, optional): Defaults to -1.0.
    """
    if Obj.ReciprocalMass == 0.0:
        return

    HalfH: float = Obj.Side_Length * 0.5
    Bottom: float = Obj.Position[1] - HalfH

    if Bottom < GroundHeight:
        Obj.Position[1] = GroundHeight + HalfH
        Vy: float = Obj.Velocity[1]

        if Vy < 0.0:
            if abs(Vy) < 0.2:
                Obj.Velocity[1] = 0.0

            else:
                Obj.Velocity[1] = -Obj.Restitution * Vy


def solve_ground_collision_sphere(Obj: P_Object, GroundHeight: float = -1.0) -> None:
    """
    Take care of the special case for the "Ground(Plane)" with sphere

    Args:
        Object (P_Object): sphere
        GroundHeight (float, optional): Defaults to -1.0.
    """
    if Obj.ReciprocalMass == 0.0:
        return

    Radius: float = Obj.Side_Length
    Bottom: float = Obj.Position[1] - Radius

    if Bottom < GroundHeight:
        Obj.Position[1] = GroundHeight + Radius
        Vy: float = Obj.Velocity[1]

        if Vy < 0.0:
            if abs(Vy) < 0.2:
                Obj.Velocity[1] = 0.0
            else:
                Obj.Velocity[1] = -Obj.Restitution * Vy


def the_collision(ObjectA: P_Object, ObjectB: P_Object) -> None:
    """
    Determine whether the collision had happened
    If happened, then move the two object based the normal and the depth of the penetration

    Args:
        ObjectA (P_Object)
        ObjectB (P_Object)
    """
    InvMass_A: float = ObjectA.ReciprocalMass
    InvMass_B: float = ObjectB.ReciprocalMass

    if not ObjectA.Collidable or not ObjectB.Collidable:
        return

    IsColliding: bool
    Simplex: list[NDArray[np.float32]]
    IsColliding, Simplex = check_collison(ObjectA, ObjectB)

    if not IsColliding:
        return

    Support_fn: Callable[[NDArray[np.float32]], NDArray[np.float32]] = (
        SimplexSolver.make_support_fn(ObjectA, ObjectB)
    )

    Normal: NDArray[np.float32] | None = None
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

    if Normal is None or Depth <= 0.0:
        return

    CenterA: NDArray[np.float32] = ObjectA.Position
    CenterB: NDArray[np.float32] = ObjectB.Position
    if np.dot(CenterB - CenterA, Normal) < 0:
        Normal = -Normal

    Vrel: NDArray[np.float32] = ObjectB.Velocity - ObjectA.Velocity
    VrelNormal: float = np.dot(Vrel, Normal)

    if VrelNormal < 0.0:
        Restitution: float = min(ObjectA.Restitution, ObjectB.Restitution)

        j: float = linear_scalar_impulse(
            Restitution,
            VrelNormal,
            InvMass_A,
            InvMass_B,
        )

        Impulse: NDArray[np.float32] = j * Normal

        if InvMass_A > 0.0:
            ObjectA.Velocity -= Impulse * InvMass_A

        if InvMass_B > 0.0:
            ObjectB.Velocity += Impulse * InvMass_B

    if abs(VrelNormal) < 0.05:
        c: float = Depth
    else:
        c = max(Depth - PENETRATION_SLOP, 0.0) * CORRECTION_PERCENTAGE

    Correction: NDArray[np.float32] = c * Normal

    if InvMass_A == 0.0:
        ObjectB.Position += Correction

    elif InvMass_B == 0.0:
        ObjectA.Position -= Correction

    else:
        TotalInvMass: float = InvMass_A + InvMass_B
        ObjectA.Position -= Correction * (InvMass_A / TotalInvMass)
        ObjectB.Position += Correction * (InvMass_B / TotalInvMass)
