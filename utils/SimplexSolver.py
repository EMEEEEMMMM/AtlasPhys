import numpy as np
from numpy.typing import NDArray
from typing import Callable

from utils.GJK import get_simplex_points
from utils.G_Object import P_Object
from utils.MathPhys_utils import normalize


def make_support_fn(
    ObjectA: P_Object, ObjectB: P_Object
) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
    def support_fn(Direction: NDArray[np.float32]) -> NDArray[np.float32]:
        return get_simplex_points(ObjectA, ObjectB, Direction)

    return support_fn


def expand_segment_to_tetra(
    Simplex: list[NDArray[np.float32]],
    Support_fn: Callable[[NDArray[np.float32]], NDArray[np.float32]],
) -> list[NDArray[np.float32]]:
    A: NDArray[np.float32]
    B: NDArray[np.float32]
    A, B = Simplex  # type: ignore
    AB = B - A

    if abs(AB[0]) < 0.9:
        Axis: NDArray[np.float32] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    else:
        Axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    N1: NDArray[np.float32] = normalize(np.cross(AB, Axis))

    C: NDArray[np.float32] = Support_fn(N1)

    if np.linalg.norm(C - A) < 1e-6:
        C = Support_fn(-N1)  # type: ignore

    N2: NDArray[np.float32] = normalize(np.cross(AB, C - A))
    D: NDArray[np.float32] = Support_fn(N2)

    return [A, B, C, D]


def expand_triangle_to_tetra(
    Simplex: NDArray[np.float32],
    Support_fn: Callable[[NDArray[np.float32]], NDArray[np.float32]],
) -> list[NDArray[np.float32]] | None:
    A: NDArray[np.float32]
    B: NDArray[np.float32]
    C: NDArray[np.float32]
    A, B, C = Simplex  # type: ignore

    AB: NDArray[np.float32] = B - A
    AC: NDArray[np.float32] = C - A

    N: NDArray[np.float32] = np.cross(AB, AC)
    Norm: np.float32 = np.linalg.norm(N)

    if Norm < 1e-6:
        return None

    N /= Norm  # type: ignore

    D1: NDArray[np.float32] = Support_fn(N)
    D2: NDArray[np.float32] = Support_fn(-N)

    if np.dot(D1, D1) > np.dot(D2, D2):
        D = D1

    else:
        D = D2  # type: ignore

    return [A, B, C, D]


def triangle_contact(Simplex: list[NDArray[np.float32]]):
    A: NDArray[np.float32]
    B: NDArray[np.float32]
    C: NDArray[np.float32]
    A, B, C = Simplex  # type: ignore

    AB = B - A
    AC = C - A

    N: NDArray[np.float32] = np.cross(AB, AC)
    Norm: np.float32 = np.linalg.norm(N)

    if Norm < 1e-6:
        return None, 0.0

    N /= Norm  # type: ignore

    if np.dot(N, A) > 0.6:
        N = -N  # type: ignore

    Depth: float = abs(np.dot(N, A))

    return N, Depth
