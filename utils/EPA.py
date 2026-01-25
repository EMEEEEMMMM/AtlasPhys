import numpy as np
from numpy.typing import NDArray
from numba import njit, jit  # type: ignore

from utils.G_Object import P_Object
from utils.GJK import get_simplex_points
from utils.MathPhys_utils import normalize


TOLERANCE: float = 1e-4
MAX_ITERATIONS: int = 32


def epa(
    Simplex: list[NDArray[np.float32]], ObjectA: P_Object, ObjectB: P_Object
) -> tuple[NDArray[np.float32], float]:
    """
    The main part of epa algorithm

    Args:
        Simplex (list[NDArray[np.float32]]): contains four point (x,y,z), the simplex after gjk detected the collision
        ObjectA (P_Object)
        ObjectB (P_Object)

    Returns:
        _type_: _description_
    """

    # The vertex indexing of the polytope
    Faces: list[list[int]] = [
        [0, 1, 2],
        [0, 3, 1],
        [2, 1, 3],
        [2, 3, 0],
    ]

    Polytope: list[NDArray[np.float32]] = Simplex

    # The initial normal and distance of the face
    Normals: list[NDArray[np.float32]]
    Min_Dist: float
    Min_Face_Idx: int
    Normals, Min_Dist, Min_Face_Idx = _get_face_normals(Polytope, Faces)

    for _ in range(MAX_ITERATIONS):
        MinNormal: NDArray[np.float32] = Normals[Min_Face_Idx]
        MinDistance: float = Min_Dist
        SupportPoint: NDArray[np.float32] = get_simplex_points(
            ObjectA, ObjectB, MinNormal
        )
        SupportDistance: float = np.dot(SupportPoint, MinNormal)

        # If abs(SupportDistance - MinDistance) < TOLERANCE,
        # we consider that the MinNormal and SupportDistance is the right answer that we want
        if abs(SupportDistance - MinDistance) < TOLERANCE:
            return MinNormal, SupportDistance

        unique_edges: list[tuple[int, int]] = []

        for i in range(len(Faces) - 1, -1, -1):
            if np.dot(Normals[i], SupportPoint - Polytope[Faces[i][0]]) > 0:
                Face: list[int] = Faces.pop(i)
                Normals.pop(i)

                _add_if_unique_edge(unique_edges, Face[0], Face[1])
                _add_if_unique_edge(unique_edges, Face[1], Face[2])
                _add_if_unique_edge(unique_edges, Face[2], Face[0])

        Idx: int = len(Polytope)
        Polytope.append(SupportPoint)

        for edge in unique_edges:
            Faces.append([edge[0], edge[1], Idx])

        Normals, MinDistance, Min_Face_Idx = _get_face_normals(Polytope, Faces)

        if Min_Face_Idx == -1:
            break

    return Normals[Min_Face_Idx], Min_Dist


def _get_face_normals(
    Polytope: list[NDArray[np.float32]], Faces: list[list[int]]
) -> tuple[list[NDArray[np.float32]], float, int]:
    """
    To find the face which is the closest to the origin

    Args:
        Polytope (list[NDArray[np.float32]]): the vertices of the current polytope
        Faces (list[list[int]]): contains the vertex indexings

    Returns:
        tuple[list[NDArray[np.float32]], float, int]: Normals, Min_Dist, Min_Face_Idx
        Normals: a list contains the normal of each face
        Min_Dist: the distance of the face closest to the origin
        Min_Face_Idx: the index of the face closest to the origin
    """
    Normals: list[NDArray[np.float32]] = []
    Min_Dist: float = float("inf")
    Min_Face_Idx: int = -1

    for idx, face in enumerate(Faces):
        Point0: NDArray[np.float32] = Polytope[face[0]]
        Point1: NDArray[np.float32] = Polytope[face[1]]
        Point2: NDArray[np.float32] = Polytope[face[2]]

        Normal: NDArray[np.float32] = normalize(
            np.cross(Point1 - Point0, Point2 - Point0)
        )
        Dist: float = np.dot(Normal, Point0)

        if Dist < 0:
            Normal = -Normal
            Dist = -Dist

        Normals.append(Normal)

        if Dist < Min_Dist:
            Min_Dist = Dist
            Min_Face_Idx = idx

    return Normals, Min_Dist, Min_Face_Idx


def _add_if_unique_edge(Edges: list[tuple[int, int]], A: int, B: int) -> None:
    ReversedEdges: tuple[int, int] = (B, A)
    if ReversedEdges in Edges:
        Edges.remove(ReversedEdges)

    else:
        Edges.append((A, B))
