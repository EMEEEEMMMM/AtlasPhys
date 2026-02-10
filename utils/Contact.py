import numpy as np
from numpy.typing import NDArray

from utils.G_Object import P_Object
from utils.GJK import check_aabb
from utils.MathPhys_utils import normalize
from utils.Step import the_collision

DT: float = 0.01
BAUMGARTE: float = 0.15
SLOP: float = 0.01
FRICTION: float = 0.5

MAX_CONTACTS: int = 4
CONTACT_MATCH_DIST: float = 0.02


class ContactPoint:
    __slots__ = (
        "LocalA",
        "LocalB",
        "Penetration",
        "NormalImpulse",
        "TangentImpulse",
        "Ra",
        "Rb",
        "Active",
    )

    def __init__(
        self,
        LocalA: NDArray[np.float32],
        LocalB: NDArray[np.float32],
        Penetration: float,
    ) -> None:
        self.LocalA: NDArray[np.float32] = LocalA
        self.LocalB: NDArray[np.float32] = LocalB
        self.Penetration: float = Penetration

        self.NormalImpulse: float = 0.0
        self.TangentImpulse: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
        self.Ra: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.Rb: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.Active: bool = True


class PersistentContactManifold:
    __slots__ = (
        "A",
        "B",
        "Normal",
        "Tangent1",
        "Tangent2",
        "Contacts",
        "Key",
    )

    def __init__(self, A: P_Object, B: P_Object):
        self.A: P_Object = A  # type: ignore
        self.B: P_Object = B  # type: ignore
        self.Normal: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.Tangent1: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.Tangent2: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.Contacts: list[ContactPoint] = []
        self.Key: tuple[int, int] = (id(A), id(B))

    def update_from_collision(
        self,
        Normal: NDArray[np.float32],
        ContactPoints_World: list[tuple[NDArray[np.float32], float]],
    ) -> None:
        if Normal.size < 3 or np.linalg.norm(Normal) < 1e-6:
            return
        
        self.Normal = normalize(Normal)

        if abs(self.Normal[1]) < 0.9:
            UpVector: NDArray[np.float32] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        else:
            UpVector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self.Tangent1 = normalize(np.cross(self.Normal, UpVector))

        if np.linalg.norm(self.Tangent1) < 1e-6:
            UpVector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.Tangent1 = normalize(np.cross(self.Normal, UpVector))

        self.Tangent2 = normalize(np.cross(self.Normal, self.Tangent1))

        NewContacts: list[ContactPoint] = []

        for p, penetration in ContactPoints_World:
            LocalA: NDArray[np.float32] = self.A.RotationMatrix.T @ (
                p - self.A.Position  # type: ignore
            )
            LocalB: NDArray[np.float32] = self.B.RotationMatrix.T @ (p - self.B.Position)  # type: ignore

            C_Point: ContactPoint = ContactPoint(LocalA, LocalB, penetration)
            NewContacts.append(C_Point)

        self._match_contacts(NewContacts)
        self._reduce_contacts()

    def _match_contacts(self, NewContacts: list[ContactPoint]) -> None:
        for nc in NewContacts:
            Best = None  # type: ignore
            BestDist: float = CONTACT_MATCH_DIST

            for oc in self.Contacts:
                Wa = self.A.RotationMatrix @ oc.LocalA + self.A.Position
                Wb = self.B.RotationMatrix @ oc.LocalB + self.B.Position
                Wp = 0.5 * (Wa + Wb)

                Na = self.A.RotationMatrix @ nc.LocalA + self.A.Position
                Nb = self.B.RotationMatrix @ nc.LocalB + self.B.Position
                Npw = 0.5 * (Na + Nb)

                Dist: float = np.linalg.norm(Wp - Npw)  # type: ignore
                if Dist < BestDist:
                    Best: ContactPoint = oc
                    BestDist: float = Dist

            if Best is not None:  # type: ignore
                nc.NormalImpulse = Best.NormalImpulse
                nc.TangentImpulse = Best.TangentImpulse.copy()

        self.Contacts = NewContacts

    def _reduce_contacts(self) -> None:
        if len(self.Contacts) <= MAX_CONTACTS:
            return

        self.Contacts.sort(key=lambda c: c.Penetration, reverse=True)
        self.Contacts = self.Contacts[:MAX_CONTACTS]

    def warm_start(self) -> None:
        for C in self.Contacts:
            C.Ra = self.A.RotationMatrix @ C.LocalA
            C.Rb = self.B.RotationMatrix @ C.LocalB

            Impulse = C.NormalImpulse * self.Normal

            Impulse += C.TangentImpulse[0] * self.Tangent1
            Impulse += C.TangentImpulse[1] * self.Tangent2

            _apply_impulse(self.A, self.B, Impulse, C.Ra, C.Rb)

    def solve(self) -> None:
        for C in self.Contacts:
            Ra = C.Ra
            Rb = C.Rb

            Va: NDArray[np.float32] = self.A.Velocity + np.cross(self.A.AngularVelocity, Ra)  # type: ignore
            Vb: NDArray[np.float32] = self.B.Velocity + np.cross(self.B.AngularVelocity, Rb)  # type: ignore
            Vrel: NDArray[np.float32] = Vb - Va
            Vn: float = np.dot(Vrel, self.Normal)

            k: float = (
                self.A.ReciprocalMass
                + self.B.ReciprocalMass
                + np.dot(
                    np.cross(self.A.InvInertiaWorld @ np.cross(Ra, self.Normal), Ra)
                    + np.cross(self.B.InvInertiaWorld @ np.cross(Rb, self.Normal), Rb),
                    self.Normal,
                )
            )

            if k == 0.0:
                continue

            j: float = -Vn / k

            Old: float = C.NormalImpulse
            C.NormalImpulse = max(Old + j, 0.0)
            Delta_j: float = C.NormalImpulse - Old

            _apply_impulse(
                self.A,
                self.B,
                Delta_j * self.Normal,
                Ra,
                Rb,
            )

            if FRICTION > 0:
                Va = self.A.Velocity + np.cross(self.A.AngularVelocity, Ra)  # type: ignore
                Vb = self.B.Velocity + np.cross(self.B.AngularVelocity, Rb)  # type: ignore
                Vrel = Vb - Va

                for i, tangent in enumerate([self.Tangent1, self.Tangent2]):
                    Vt: float = np.dot(Vrel, tangent)

                    kt: float = (
                        self.A.ReciprocalMass
                        + self.B.ReciprocalMass
                        + np.dot(
                            np.cross(self.A.InvInertiaWorld @ np.cross(Ra, tangent), Ra)
                            + np.cross(
                                self.B.InvInertiaWorld @ np.cross(Rb, tangent), Rb
                            ),
                            tangent,
                        )
                    )

                    if kt == 0.0:
                        continue

                    jt: float = -Vt / kt

                    MaxFriction: float = FRICTION * C.NormalImpulse
                    OldTangent = C.TangentImpulse[i]
                    C.TangentImpulse[i] = np.clip(
                        OldTangent + jt, -MaxFriction, MaxFriction
                    )
                    Delta_jt = C.TangentImpulse[i] - OldTangent

                    _apply_impulse(self.A, self.B, Delta_jt * tangent, Ra, Rb)

    def positional_correction(self) -> None:
        if self.A.ReciprocalMass + self.B.ReciprocalMass == 0:
            return

        Penetration: float = max(c.Penetration for c in self.Contacts)
        if Penetration <= SLOP:
            return

        Correction: NDArray[np.float32] = (
            0.2
            * (Penetration - SLOP)
            / (self.A.ReciprocalMass + self.B.ReciprocalMass)
            * self.Normal
        )

        if self.A.ReciprocalMass > 0:
            self.A.Position -= Correction * self.A.ReciprocalMass
        if self.B.ReciprocalMass > 0:
            self.B.Position += Correction * self.B.ReciprocalMass


class ManifoldManager:
    def __init__(self) -> None:
        self.Manifolds: dict[tuple[int, int], PersistentContactManifold] = {}

    def update(self, Objects: list[P_Object]) -> list[PersistentContactManifold]:
        ActiveKeys: set[int] = set()
        Result: list[PersistentContactManifold] = []

        N: int = len(Objects)
        for i in range(N):
            for j in range(i + 1, N):
                A: P_Object = Objects[i]
                B: P_Object = Objects[j]

                if not check_aabb(
                    A.X_MAX + A.Position[0],
                    A.X_MIN + A.Position[0],
                    A.Y_MAX + A.Position[1],
                    A.Y_MIN + A.Position[1],
                    A.Z_MAX + A.Position[2],
                    A.Z_MIN + A.Position[2],
                    B.X_MAX + B.Position[0],
                    B.X_MIN + B.Position[0],
                    B.Y_MAX + B.Position[1],
                    B.Y_MIN + B.Position[1],
                    B.Z_MAX + B.Position[2],
                    B.Z_MIN + B.Position[2],
                ):
                    continue

                Key: tuple[int, int] = (id(A), id(B))
                ActiveKeys.add(Key)  # type: ignore

                if Key not in self.Manifolds:
                    self.Manifolds[Key] = PersistentContactManifold(A, B)

                Manifold: PersistentContactManifold = self.Manifolds[Key]

                ContactsWorld, Normal = generate_contacts(A, B)  # type: ignore
                if not ContactsWorld:
                    continue

                Manifold.update_from_collision(Normal, ContactsWorld)
                Result.append(Manifold)

        Dead: list[tuple[int, int]] = [
            k for k in self.Manifolds.keys() if k not in ActiveKeys  # type: ignore
        ]
        for k in Dead:
            del self.Manifolds[k]

        return Result


def generate_contacts(
    A: P_Object, B: P_Object
) -> tuple[list[tuple[NDArray[np.float32], float]] | None, NDArray[np.float32]]:
    """
    To generate contacts for the following types of objects: Plane, Sphere, Cube, Convex(later)

    Args:
        A (P_Object)
        B (P_Object)

    Returns:
        list[Contact] | None: a list containing all the contacts
    """
    ShapeA: str = A.Shape
    ShapeB: str = B.Shape

    match ShapeA, ShapeB:
        case "Plane", "Cube":
            return generate_pc_contacts(A, B)

        case "Cube", "Plane":
            return generate_pc_contacts(B, A)

        case "Plane", "Sphere":
            return generate_ps_contacts(A, B)

        case "Sphere", "Plane":
            return generate_ps_contacts(B, A)

        case "Sphere", "Sphere":
            return generate_ss_contacts(A, B)

        case "Cube", "Cube":
            return generate_cc_contacts(A, B)

        case "Cube", "Sphere":
            return generate_cs_contacts(A, B)

        case "Sphere", "Cube":
            Contacts, Normal = generate_cs_contacts(B, A)
            return Contacts, -Normal

        case _:
            return None, np.array([], dtype=np.float32)


def generate_pc_contacts(
    Plane: P_Object,
    Cube: P_Object,
) -> tuple[list[tuple[NDArray[np.float32], float]], NDArray[np.float32]]:
    """
    Generate contacts of the plane and the cube

    Args:
        Plane (P_Object)
        Cube (P_Object)

    Returns:
        list[Contact]: a list containing all the contacts
    """
    Normal: NDArray[np.float32] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    PlaneHeight: float = Plane.Position[1]
    Contacts: list[tuple[NDArray[np.float32], float]] = []

    for vertex in Cube.XYZVertices:
        VertexWorld: NDArray[np.float32] = Cube.RotationMatrix @ vertex + Cube.Position
        Dist: float = np.dot(Normal, VertexWorld) - PlaneHeight

        if Dist < 0.0:
            Penetration: float = -Dist
            Contacts.append((VertexWorld, Penetration))

    return Contacts, Normal


def generate_ps_contacts(
    Plane: P_Object, Sphere: P_Object
) -> tuple[list[tuple[NDArray[np.float32], float]], NDArray[np.float32]]:
    """
    Generate contacts of the plane and the sphere

    Args:
        Plane (P_Object)
        Sphere (P_Object)

    Returns:
        list[Contact]: a list containing all the contacts
    """
    Normal: NDArray[np.float32] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    Height: float = Plane.Position[1]

    Dist: float = np.dot(Normal, Sphere.Position) - Height

    if Dist >= Sphere.Side_Length:
        return [], Normal

    Point: NDArray[np.float32] = Sphere.Position - Normal * Sphere.Side_Length  # type: ignore
    Penetration: float = Sphere.Side_Length - Dist

    return [(Point, Penetration)], Normal


def generate_ss_contacts(
    SphereA: P_Object, SphereB: P_Object
) -> tuple[list[tuple[NDArray[np.float32], float]], NDArray[np.float32]]:
    """
    Generate contacts of two spheres

    Args:
        SphereA
        SphereB

    Returns:
        list[Contact]: a list containing all the contacts
    """
    d: NDArray[np.float32] = SphereB.Position - SphereA.Position
    Dist: np.float32 = np.linalg.norm(d)

    RadiusSum: float = SphereA.Side_Length + SphereB.Side_Length
    if Dist >= RadiusSum:
        return [], np.array([], dtype=np.float32)

    Normal: NDArray[np.float32] = d / (Dist + 1e-8)
    Point: NDArray[np.float32] = SphereA.Position + Normal * SphereA.Side_Length  # type: ignore
    Penetration: float = RadiusSum - Dist  # type: ignore

    return [(Point, Penetration)], Normal  # type: ignore


def generate_cc_contacts(
    CubeA: P_Object, CubeB: P_Object
) -> tuple[list[tuple[NDArray[np.float32], float]], NDArray[np.float32]]:
    IsColliding, Normal, DepthEPA = the_collision(CubeA, CubeB)
    if not IsColliding:
        return [], np.array([], dtype=np.float32)

    if np.dot(Normal, CubeB.Position - CubeA.Position) < 0:
        Normal = -Normal

    Na: NDArray[np.float32]
    Ca: NDArray[np.float32]
    Facea: list[NDArray[np.float32]]
    Nb: NDArray[np.float32]
    Cb: NDArray[np.float32]
    Faceb: list[NDArray[np.float32]]
    Na, Ca, Facea = select_reference_face(CubeA, Normal)
    Nb, Cb, Faceb = select_reference_face(CubeB, -Normal)

    if abs(np.dot(Na, Normal)) > abs(np.dot(Nb, -Normal)):
        RefNormal: NDArray[np.float32] = Na
        RefCenter: NDArray[np.float32] = Ca
        RefFace: list[NDArray[np.float32]] = Facea
        IncFace: list[NDArray[np.float32]] = Faceb

    else:
        RefNormal: NDArray[np.float32] = -Nb
        RefCenter: NDArray[np.float32] = Cb
        RefFace: list[NDArray[np.float32]] = Faceb
        IncFace: list[NDArray[np.float32]] = Facea
        Normal = -Normal

    Poly: list[NDArray[np.float32]] = IncFace.copy()

    for i in range(4):
        a: NDArray[np.float32] = RefFace[i]
        b: NDArray[np.float32] = RefFace[(i + 1) % 4]
        Edge: NDArray[np.float32] = b - a
        PlaneN: NDArray[np.float32] = normalize(np.cross(Edge, RefNormal))
        PlaneD: float = np.dot(PlaneN, a)

        if np.dot(PlaneN, RefCenter) - PlaneD < 0:
            PlaneN = -PlaneN
            PlaneD = np.dot(PlaneN, a)

        Poly = clip_polygon(Poly, PlaneN, PlaneD)
        if not Poly:
            return [], Normal

    Contacts: list[tuple[NDArray[np.float32], float]] = []

    for p in Poly:
        Contacts.append((p, DepthEPA))

    return Contacts, Normal


def generate_cs_contacts(
    Cube: P_Object, Sphere: P_Object
) -> tuple[list[tuple[NDArray[np.float32], float]], NDArray[np.float32]]:
    R: NDArray[np.float32] = Cube.RotationMatrix
    PLocal: NDArray[np.float32] = R.T @ (Sphere.Position - Cube.Position)  # type: ignore

    HalfExtent: NDArray[np.float32] = Cube.HalfExtent
    ClosestLocal: NDArray[np.float32] = np.clip(PLocal, -HalfExtent, HalfExtent)
    ClosestWorld: NDArray[np.float32] = Cube.Position + R @ ClosestLocal  # type: ignore

    Delta: NDArray[np.float32] = Sphere.Position - ClosestWorld
    Dist: float = np.linalg.norm(Delta)  # type: ignore

    Distances: NDArray[np.float32] = np.array([
        HalfExtent[0] - abs(PLocal[0]),
        HalfExtent[1] - abs(PLocal[1]),
        HalfExtent[2] - abs(PLocal[2]),
    ], dtype=np.float32)

    MinAxis: int = np.argmin(Distances)  # type: ignore
    Normal = np.zeros(3, dtype=np.float32)
    Normal[MinAxis] = 1.0 if PLocal[MinAxis] > 0 else -1.0

    Normal = normalize(R @ Normal)

    Penetration: float = Sphere.Side_Length + Distances[MinAxis]
    ContactLocal = PLocal.copy()
    ContactLocal[MinAxis] = HalfExtent[MinAxis] * Normal[MinAxis]
    ContactWorld = Cube.Position + R @ ContactLocal

    return [(ContactWorld, Penetration)], Normal  # type: ignore


def _apply_impulse(
    A: P_Object,
    B: P_Object,
    J: NDArray[np.float32],
    Ra: NDArray[np.float32],
    Rb: NDArray[np.float32],
) -> None:
    if A.ReciprocalMass > 0:
        A.Velocity -= J * A.ReciprocalMass
        A.AngularVelocity -= A.InvInertiaWorld @ np.cross(Ra, J)

    if B.ReciprocalMass > 0:
        B.Velocity += J * B.ReciprocalMass
        B.AngularVelocity += B.InvInertiaWorld @ np.cross(Rb, J)


def select_reference_face(
    Cube: P_Object, Normal: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[NDArray[np.float32]]]:
    """
    Find the most aligned one with the normal in all six faces of the cube

    Args:
        Cube (P_Object)
        Normal (NDArray[np.float32])

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32], list[NDArray[np.float32]]]
        1. Face normal
        2. Face center
        3. Face vertices
    """
    LocalNormals: list[NDArray[np.float32]] = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    ]

    BestDot: float = -1.0
    BestFace: NDArray[np.float32] = np.array([], dtype=np.float32)

    for ln in LocalNormals:
        Wn: NDArray[np.float32] = Cube.RotationMatrix @ ln
        d: float = np.dot(Wn, Normal)
        if d > BestDot:
            BestDot = d
            BestFace = ln

    HalfExtent: NDArray[np.float32] = Cube.HalfExtent
    Center: NDArray[np.float32] = Cube.Position + Cube.RotationMatrix @ (BestFace * HalfExtent)  # type: ignore

    Axes: list[int] = [0, 1, 2]
    Axes.remove(np.argmax(np.abs(BestFace)))  # type: ignore

    Vertices: list[NDArray[np.float32]] = []
    SignPermutations: list[tuple[int, int]] = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    for s1, s2 in SignPermutations:
        v: NDArray[np.float32] = BestFace * HalfExtent
        v[Axes[0]] = s1 * HalfExtent[Axes[0]]
        v[Axes[1]] = s2 * HalfExtent[Axes[1]]
        Vertices.append(Cube.RotationMatrix @ v + Cube.Position)

    return Cube.RotationMatrix @ BestFace, Center, Vertices


def clip_polygon(
    Poly: list[NDArray[np.float32]], PlaneN: NDArray[np.float32], PlaneD: float
) -> list[NDArray[np.float32]]:
    Out: list[NDArray[np.float32]] = []
    for i in range(len(Poly)):
        A: NDArray[np.float32] = Poly[i]
        B: NDArray[np.float32] = Poly[(i + 1) % len(Poly)]

        Da: float = np.dot(PlaneN, A) - PlaneD
        Db: float = np.dot(PlaneN, B) - PlaneD

        if Da >= 0:
            Out.append(A)

        if Da * Db < 0:
            t: float = Da / (Da - Db)
            Out.append(A + t * (B - A))  # type: ignore

    return Out
