from numpy.typing import NDArray
import numpy as np
from utils.G_Object import P_Object
from utils.MathPhys_utils import normalize


def _support_function_polygon(
    ObjectA: P_Object, Direction: NDArray[np.float32]
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
    
    ModelMatrix: NDArray[np.float32] = ObjectA.get_model_matrix()

    Direction4D: NDArray[np.float32] = np.append(Direction, 0.0)
    
    LocalDirection = np.dot(Direction4D, ModelMatrix)[:3]   # * only need x,y,z

    Vertices: NDArray[np.float32] = ObjectA.Vertices.reshape(-1, 7)[:, :3]

    idx: np.intp = np.argmax(np.dot(Vertices, LocalDirection))

    BestLocalPoint: NDArray[np.float32] = Vertices[idx]

    Point4D: NDArray[np.float32] = np.append(BestLocalPoint, 1.0)

    BestWorldPoint: NDArray[np.float32] = np.dot(ModelMatrix, Point4D)[:3]   # * only need x,y,z

    return BestWorldPoint  # type: ignore


def _get_simplex_points(
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
    PointA: NDArray[np.float32] = _support_function_polygon(ShapeA, Direction)
    PointB: NDArray[np.float32] = _support_function_polygon(ShapeB, -Direction)

    return PointA - PointB


def _handle_simplex(
    Simplex: list[NDArray[np.float32]], Direction: NDArray[np.float32]
) -> bool:
    """
    Determines if the simplex contains the origin and updates the direction.

    Args:
        Simplex (NDArray[NDArray[np.float32]]): The simplex (points are from _get_simplex_points)
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


def _linear_case(Simplex: list[NDArray[np.float32]], Direction: NDArray[np.float32]) -> bool:
    """
    The case when the simplex is a straight line (contains two points)

    Args:
        Simplex (NDArray[np.float32]): Simplex = [B, A] A is the lastest
        Direction (NDArray[np.float32])

    Returns:
        bool: The result whether had collision
    """
    B: float
    A: float

    B, A = Simplex  # type: ignore
    AB: float = B - A
    AO: float = -A

    if np.dot(AB, AO) > 0:
        Direction[:] = np.cross(np.cross(AB, AO), AB)

    else:
        del Simplex[0]
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
        bool: The result whether had collision
    """
    A: float
    B: float
    C: float

    C, B, A = Simplex  # type:ignore

    AB: float = B - A
    AC: float = C - A
    AO: float = -A

    ABC: NDArray[np.float32] = np.cross(AB, AC)

    if np.dot(np.cross(ABC, AC), AO) > 0:
        if np.dot(AC, AO) > 0:
            del Simplex[1]
            Direction[:] = np.cross(np.cross(AC, AO), AC)
        
        else: 
            del Simplex[0]
            return _linear_case(Simplex, Direction)
        
    else: 
        if np.dot(np.cross(AB, ABC), AO) > 0:
            del Simplex[0]
            return _linear_case(Simplex, Direction)
        
        else:
            if np.dot(ABC, AO) > 0:
                Direction[:] = ABC

            else:
                Direction[:] = -ABC
                Simplex[0], Simplex[1] = Simplex[1], Simplex[0]

            return False
        
    return False

def _tetrahedron_case(Simplex: list[NDArray[np.float32]], Direction: NDArray[np.float32]) -> bool:
    """
    The case when the simplex is a tetrahedron (contains four points)

    Args:
        Simplex (NDArray[np.float32]): Simplex = [D, C, B, A] A is the lastest
        Direction (NDArray[np.float32])

    Returns:
        bool: The result whether had collision
    """
    A: float; B: float; C: float; D: float

    D, C, B, A = Simplex   # type: ignore
    AB: float = B - A
    AC: float = C - A
    AD: float = D - A
    AO: float = -A

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
        del Simplex[1]
        return _triangle_case(Simplex, Direction)
    
    return True

def check_collison(ObjectA: P_Object, ObjectB: P_Object) -> bool:
    """
    The main function to detect whether two obejcts had collision

    Args:
        ObjectA (P_Object)
        ObjectB (P_Object)

    Returns:
        bool: whether the two objects collide
    """

    Direction: NDArray[np.float32] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    Simplex: list[NDArray[np.float32]] = [_get_simplex_points(ObjectA, ObjectB, Direction)]

    Direction = -Simplex[0]

    while True:
        
        A: NDArray[np.float32] = _get_simplex_points(ObjectA, ObjectB, Direction)

        if np.dot(A, Direction) < 0:
            return False
        
        Simplex.append(A)

        if _handle_simplex(Simplex, Direction):
            print("True")
            return True