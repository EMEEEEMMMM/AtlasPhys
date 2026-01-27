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
) -> tuple[NDArray[np.float32] | None, float]:
    """
    The main part of epa algorithm

    Args:
        Simplex (list[NDArray[np.float32]]): contains four point (x,y,z), the simplex after gjk detected the collision
        ObjectA (P_Object)
        ObjectB (P_Object)

    Returns:
        _type_: _description_
    """
    if len(Simplex) < 4:
        return None, 0.0

    # The vertex indexing of the polytope
    Faces: list[list[int]] = [
        [0, 1, 2],
        [0, 3, 1],
        [0, 2, 3],
        [1, 3, 2],
    ]

    Polytope: list[NDArray[np.float32]] = Simplex.copy()

    # The initial normals and distances of the face
    Normals: list[NDArray[np.float32]]
    Distances: list[float]
    Normals, Distances = _initialize_polytope(Polytope, Faces)

    Min_Face_Idx: int = int(np.argmin(Distances))
    MinDist: float = Distances[Min_Face_Idx]

    Iteration: int = 0

    while Iteration < MAX_ITERATIONS:

        Iteration += 1

        MinNormal: NDArray[np.float32] = Normals[Min_Face_Idx]
        MinDistance: float = MinDist

        SupportPoint: NDArray[np.float32] = get_simplex_points(
            ObjectA, ObjectB, MinNormal
        )
        SupportDistance: float = np.dot(SupportPoint, MinNormal)

        # If abs(SupportDistance - MinDistance) < TOLERANCE,
        # we consider that the MinNormal and SupportDistance is the right answer that we want
        if SupportDistance - MinDistance <= TOLERANCE:
            return MinNormal, MinDistance

        if Iteration > MAX_ITERATIONS - 2:
            return MinNormal, MinDistance

        unique_edges: list[tuple[int, int]] = []

        i: int = len(Faces) - 1
        while i >= 0:
            FacePoint: NDArray[np.float32] = Polytope[Faces[i][0]]
            if np.dot(Normals[i], SupportPoint - FacePoint) > 0:
                RemovedFace: list[int] = Faces.pop(i)
                Normals.pop(i)
                Distances.pop(i)

                _add_if_unique_edge(unique_edges, RemovedFace[0], RemovedFace[1])
                _add_if_unique_edge(unique_edges, RemovedFace[1], RemovedFace[2])
                _add_if_unique_edge(unique_edges, RemovedFace[2], RemovedFace[0])

            i -= 1

        NewVertexIdx: int = len(Polytope)
        Polytope.append(SupportPoint)

        NewNormals: list[NDArray[np.float32]] = []
        NewDistances: list[float] = []

        for edge in unique_edges:
            NewFaces: list[int] = [edge[0], edge[1], NewVertexIdx]
            Faces.append(NewFaces)

            Normal, Distance = _get_face_normal(
                Polytope[NewFaces[0]], Polytope[NewFaces[1]], Polytope[NewFaces[2]]
            )

            NewNormals.append(Normal)
            NewDistances.append(Distance)

        Normals.extend(NewNormals)
        Distances.extend(NewDistances)

        Min_Face_Idx = int(np.argmin(Distances))
        MinDist = Distances[Min_Face_Idx]

    return Normals[Min_Face_Idx], MinDist


def _initialize_polytope(
    Polytope: list[NDArray[np.float32]], Faces: list[list[int]]
) -> tuple[list[NDArray[np.float32]], list[float]]:
    """
    Initialize the polytope by calculating normals and distances for all faces.

    Args:
        Polytope (list[NDArray[np.float32]])
        Faces (list[list[int]])

    Returns:
        tuple[list[NDArray[np.float32]], list[float]]
    """
    Normals: list[NDArray[np.float32]] = []
    Distances: list[float] = []

    for face in Faces:
        Point0: NDArray[np.float32] = Polytope[face[0]]
        Point1: NDArray[np.float32] = Polytope[face[1]]
        Point2: NDArray[np.float32] = Polytope[face[2]]

        Normal, Distance = _get_face_normal(Point0, Point1, Point2)
        Normals.append(Normal)
        Distances.append(Distance)

    return Normals, Distances


def _get_face_normal(
    Point0: NDArray[np.float32],
    Point1: NDArray[np.float32],
    Point2: NDArray[np.float32],
) -> tuple[NDArray[np.float32], float]:
    """
    Calculate the outward-pointing normal and distance to origin for a triangular face.

    The normal should point away from the polytope interior (where origin is)
    Distance is the perpendicular distance from origin to the face plane.

    Args:
        Point0 (NDArray[np.float32])
        Point1 (NDArray[np.float32])
        Point2 (NDArray[np.float32])

    Returns:
        tuple[Normal: NDArray[np.float32], Distance: np.float32]
    """

    Edge1: NDArray[np.float32] = Point1 - Point0
    Edge2: NDArray[np.float32] = Point2 - Point0

    Normal: NDArray[np.float32] = normalize(np.cross(Edge1, Edge2))

    Distance: float = np.dot(Normal, Point0)

    if Distance < 0:
        Normal = -Normal
        Distance = -Distance

    return Normal, Distance


def _add_if_unique_edge(Edges: list[tuple[int, int]], A: int, B: int) -> None:
    ReversedEdges: tuple[int, int] = (B, A)
    if ReversedEdges in Edges:
        Edges.remove(ReversedEdges)

    else:
        Edges.append((A, B))
