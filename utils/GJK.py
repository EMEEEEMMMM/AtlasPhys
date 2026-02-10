from numpy.typing import NDArray
import numpy as np
from numba import njit  # type: ignore

from utils.G_Object import P_Object
from utils.MathPhys_utils import normalize


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def _support_function_polygon(
    Direction: NDArray[np.float32],
    ModelMatrix: NDArray[np.float32],
    XYZVertices: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    The support function of gjk

    Args:
        ObjectA (P_Object): the object
        Direction (NDArray[np.float32]): the direction you want to chose

    Returns:
        NDArray[np.float32]: the furtherst point in the direction on the object
    """

    """
    ! How this function works
    
    1. Get the object's model matrix
    The matrix contains the object's current # ! Position, Rotation and Scale
    
    2. Convert the direction from the World Space -> Local Space
    Create a 4D vector for the direction (w = 0 means that it is a direction instead of a point)
    Multiply the direction by the tranpose of the matrix

    Local direction = Direction * Transpose of matrix
    
    3. Find the furthest point in local direction

    4. Convert the point from Local Space -> World Space
    Create a 4D vector for the point(w = 1 means the it is a point instead of a direction)
    World Point = Matrix * Point
    """
    
    InvModel: NDArray[np.float32] = np.linalg.inv(ModelMatrix)
    LocalDirection = np.dot(InvModel[:3, :3].T, Direction)  # * only need x,y,z

    Vertices: NDArray[np.float32] = XYZVertices

    idx: np.intp = np.argmax(np.dot(Vertices, LocalDirection))

    BestLocalPoint: NDArray[np.float32] = Vertices[idx]

    Point4D: NDArray[np.float32] = np.append(BestLocalPoint, np.float32(1.0))

    BestWorldPoint: NDArray[np.float32] = np.dot(ModelMatrix, Point4D)[
        :3
    ]  # * only need x,y,z

    return BestWorldPoint  # type: ignore


def get_simplex_points(
    ShapeA: P_Object, ShapeB: P_Object, Direction: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    To get the point in order to form the simplex

    Args:
        ShapeA (P_Object): The object A
        ShapeB (P_Object): The object B
        Direction (NDArray[np.float32])

    Returns:
        NDArray[np.float32]: The coordinate of the final point on simplex
    """
    PointA: NDArray[np.float32] = _support_function_polygon(
        Direction, ShapeA.get_model_matrix(), ShapeA.XYZVertices
    )
    PointB: NDArray[np.float32] = _support_function_polygon(
        -Direction, ShapeB.get_model_matrix(), ShapeB.XYZVertices
    )

    return PointA - PointB


def _handle_simplex(
    Simplex: list[NDArray[np.float32]], Direction: NDArray[np.float32]
) -> bool:
    """
    Determines if the simplex contains the origin and updates the direction.

    Args:
        Simplex (NDArray[NDArray[np.float32]]): The simplex (points are from get_simplex_points)
        Direction (NDArray[np.float32]): the direction

    Returns:
        bool: True if collision between # ! The two objects is found
              False if no collision between the two objects is found
    """
    match len(Simplex):
        case 2:
            return _linear_case(Simplex, Direction)

        case 3:
            return _triangle_case(Simplex, Direction)

        case 4:
            return _tetrahedron_case(Simplex, Direction)

        case _:
            return False


def _linear_case(
    Simplex: list[NDArray[np.float32]], Direction: NDArray[np.float32]
) -> bool:
    """
    The case when the simplex is a straight line (contains two points)

    Args:
        Simplex (NDArray[np.float32]): Simplex = [B, A] A is the lastest
        Direction (NDArray[np.float32])

    Returns:
        bool
    """
    B: NDArray[np.float32]
    A: NDArray[np.float32]

    B, A = Simplex  # type: ignore
    AB: NDArray[np.float32] = B - A
    AO: NDArray[np.float32] = -A
    
    if np.dot(AB, AO) > 0:
        Direction[:] = np.cross(np.cross(AB, AO), AB)
    
    else:
        del Simplex[1]
        Direction[:] = AO

    return False


def _triangle_case(
    Simplex: list[NDArray[np.float32]], Direction: NDArray[np.float32]
) -> bool:
    """
    The case when the simplex is a triangle (contains three points)

    Args:
        Simplex (NDArray[np.float32]): Simplex = [C, B, A] A is the lastest
        Direction (NDArray[np.float32])

    Returns:
        bool: False (since triangle cannot contain origin in 3D space)
    """
    A: NDArray[np.float32]
    B: NDArray[np.float32]
    C: NDArray[np.float32]

    C, B, A = Simplex  # type:ignore

    AB: NDArray[np.float32] = B - A
    AC: NDArray[np.float32] = C - A
    AO: NDArray[np.float32] = -A

    ABC: NDArray[np.float32] = np.cross(AB, AC)
    ACBB: NDArray[np.float32] = -np.cross(ABC, AB)
    ABCC: NDArray[np.float32] = np.cross(ABC, AC)

    if np.dot(ABCC, AO) > 0.0:
        del Simplex[1]
        Direction[:] = normalize(ABCC)

    elif np.dot(ACBB, AO) > 0.0:
        del Simplex[0]
        Direction[:] = normalize(ACBB)
    else:
        ABCO: float = np.dot(ABC, AO)
        if ABCO > 0.0:
            Direction[:] = ABC
        elif ABCO < 0.0:
            Simplex.clear()
            Simplex.extend([B, C, A])
            Direction[:] = -ABC
        else:
            return True

    return False


def _tetrahedron_case(
    Simplex: list[NDArray[np.float32]], Direction: NDArray[np.float32]
) -> bool:
    """
    The case when the simplex is a tetrahedron (contains four points)

    Args:
        Simplex (NDArray[np.float32]): Simplex = [D, C, B, A] A is the lastest
        Direction (NDArray[np.float32])

    Returns:
        bool: True if origin is detected in the tetrahedron
              False otherwise
    """
    A: NDArray[np.float32]
    B: NDArray[np.float32]
    C: NDArray[np.float32]
    D: NDArray[np.float32]

    D, C, B, A = Simplex  # type: ignore
    AB: NDArray[np.float32] = B - A
    AC: NDArray[np.float32] = C - A
    AD: NDArray[np.float32] = D - A
    AO: NDArray[np.float32] = -A

    ABC: NDArray[np.float32] = np.cross(AB, AC)
    ACD: NDArray[np.float32] = np.cross(AC, AD)
    ADB: NDArray[np.float32] = np.cross(AD, AB)

    if np.dot(ABC, AO) > 0:
        del Simplex[0]
        return _triangle_case(Simplex, Direction)

    if np.dot(ACD, AO) > 0:
        del Simplex[2]
        return _triangle_case(Simplex, Direction)

    if np.dot(ADB, AO) > 0:
        Simplex.clear()
        Simplex.extend([B, D, A])
        return _triangle_case(Simplex, Direction)

    return True


def check_collison(
    ObjectA: P_Object, ObjectB: P_Object
) -> tuple[bool, list[NDArray[np.float32]]]:
    """
    The main function to detect whether two obejcts had collision

    Args:
        ObjectA (P_Object)
        ObjectB (P_Object)

    Returns:
        bool: whether the two objects collide
    """

    Direction: NDArray[np.float32] = ObjectB.Position - ObjectA.Position

    if np.linalg.norm(Direction) < 1e-3:
        Direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    Direction = Direction / np.linalg.norm(Direction)  # type: ignore

    Simplex: list[NDArray[np.float32]] = [
        get_simplex_points(ObjectA, ObjectB, Direction)
    ]

    Direction = -Simplex[0]

    MaxIterations: int = max(len(ObjectA.XYZVertices), len(ObjectB.XYZVertices))

    while MaxIterations > 0:
        MaxIterations -= 1

        A: NDArray[np.float32] = get_simplex_points(ObjectA, ObjectB, Direction)

        if np.dot(A, Direction) < 0:
            return False, Simplex

        Simplex.append(A)

        if _handle_simplex(Simplex, Direction):
            return True, Simplex

    return False, Simplex


def check_aabb(
    X_MAX_A: float,
    X_MIN_A: float,
    Y_MAX_A: float,
    Y_MIN_A: float,
    Z_MAX_A: float,
    Z_MIN_A: float,
    X_MAX_B: float,
    X_MIN_B: float,
    Y_MAX_B: float,
    Y_MIN_B: float,
    Z_MAX_B: float,
    Z_MIN_B: float,
) -> bool:
    """
    A simple AABB detection to support the gjk algorithm

    Args:
        ObjectA (P_Object)
        ObjectB (P_Object)

    Returns:
        bool
    """
    if X_MAX_A < X_MIN_B or X_MIN_A > X_MAX_B:
        return False

    if Y_MAX_A < Y_MIN_B or Y_MIN_A > Y_MAX_B:
        return False

    if Z_MAX_A < Z_MIN_B or Z_MIN_A > Z_MAX_B:
        return False

    return True
