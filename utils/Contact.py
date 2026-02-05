import numpy as np
from numpy.typing import NDArray

from utils.G_Object import P_Object
from utils.MathPhys_utils import normalize
from utils.Step import the_collision

SLOP: float = 0.01
BETA: float = 0.2
DT: float = 0.01


class Contact:
    __slots__ = (
        "ObjectA",
        "ObjectB",
        "Point",
        "Normal",
        "Penetration",
        "AccmulatedNormalImpulse",
    )

    def __init__(
        self,
        ObjectA: P_Object,
        ObjectB: P_Object,
        Point: NDArray[np.float32],
        Normal: NDArray[np.float32],
        Penetration: float,
    ) -> None:
        self.ObjectA: P_Object = ObjectA
        self.ObjectB: P_Object = ObjectB
        self.Point: NDArray[np.float32] = Point
        self.Normal: NDArray[np.float32] = Normal
        self.Penetration: float = Penetration
        self.AccmulatedNormalImpulse: float = 0.0


def generate_contacts(A: P_Object, B: P_Object) -> list[Contact] | None:
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
            IsColliding: bool
            Normal: NDArray[np.float32]
            Depth: float
            IsColliding, Normal, Depth = the_collision(A, B)
            if not IsColliding:
                return []
            return generate_cc_contacts(A, B, Normal, Depth)

        case "Cube", "Sphere":
            return generate_cs_contacts(A, B)

        case "Sphere", "Cube":
            return generate_cs_contacts(B, A)

        case _:
            pass


def generate_pc_contacts(
    Plane: P_Object,
    Cube: P_Object,
) -> list[Contact]:
    """
    Generate contacts of the plane and the cube

    Args:
        Plane (P_Object)
        Cube (P_Object)

    Returns:
        list[Contact]: a list containing all the contacts
    """

    Contacts: list[Contact] = []

    Normal: NDArray[np.float32] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    Penetration: float = Plane.Position[1]

    for vertex in Cube.XYZVertices:
        VertexWorld: NDArray[np.float32] = Cube.RotationMatrix @ vertex + Cube.Position
        Dist: float = np.dot(Normal, VertexWorld) - Penetration

        if Dist < 0.0:
            Contacts.append(
                Contact(
                    ObjectA=Plane,
                    ObjectB=Cube,
                    Point=VertexWorld,
                    Normal=Normal,
                    Penetration=-Dist,
                )
            )

    Contacts.sort(key=lambda c: c.Penetration, reverse=True)
    return Contacts[:2]


def generate_ps_contacts(Plane: P_Object, Sphere: P_Object) -> list[Contact]:
    """
    Generate contacts of the plane and the sphere

    Args:
        Plane (P_Object)
        Sphere (P_Object)

    Returns:
        list[Contact]: a list containing all the contacts
    """
    Contacts: list[Contact] = []

    Normal: NDArray[np.float32] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    Height: float = Plane.Position[1]

    Dist: float = np.dot(Normal, Sphere.Position) - Height

    if Dist >= Sphere.Side_Length:
        return Contacts

    Point: NDArray[np.float32] = Sphere.Position - Normal * Sphere.Side_Length  # type: ignore
    Penetration: float = Sphere.Side_Length - Dist

    Contacts.append(
        Contact(
            ObjectA=Plane,
            ObjectB=Sphere,
            Point=Point,
            Normal=Normal,
            Penetration=Penetration,
        )
    )

    return Contacts


def generate_ss_contacts(SphereA: P_Object, SphereB: P_Object) -> list[Contact]:
    """
    Generate contacts of two spheres

    Args:
        SphereA
        SphereB

    Returns:
        list[Contact]: a list containing all the contacts
    """
    Contacts: list[Contact] = []

    d: NDArray[np.float32] = SphereB.Position - SphereA.Position
    Dist: np.float32 = np.linalg.norm(d)

    RadiusSum: float = SphereA.Side_Length + SphereB.Side_Length
    if Dist >= RadiusSum:
        return Contacts

    Normal: NDArray[np.float32] = d / (Dist + 1e-8)
    Point: NDArray[np.float32] = SphereA.Position + Normal * SphereA.Side_Length  # type: ignore

    Penetration: float = RadiusSum - Dist  # type: ignore

    Contacts.append(
        Contact(
            ObjectA=SphereA,
            ObjectB=SphereB,
            Point=Point,
            Normal=Normal,  # type: ignore
            Penetration=Penetration,
        )
    )

    return Contacts


def generate_cc_contacts(
    CubeA: P_Object, CubeB: P_Object, Normal: NDArray[np.float32], Penetration: float
) -> list[Contact]:
    Na: NDArray[np.float32]
    Ca: NDArray[np.float32]
    Facea: list[NDArray[np.float32]]
    Nb: NDArray[np.float32]
    Cb: NDArray[np.float32]
    Faceb: list[NDArray[np.float32]]
    Na, Ca, Facea = select_reference_face(CubeA, Normal)
    Nb, Cb, Faceb = select_reference_face(CubeB, Normal)

    if abs(np.dot(Na, Normal)) > abs(np.dot(Nb, Normal)):
        RefNormal: NDArray[np.float32] = Na
        RefCenter: NDArray[np.float32] = Ca
        RefFace: list[NDArray[np.float32]] = Facea
        IncFace: list[NDArray[np.float32]] = Faceb
        Flip: bool = False

    else:
        RefNormal: NDArray[np.float32] = Nb
        RefCenter: NDArray[np.float32] = Cb
        RefFace: list[NDArray[np.float32]] = Faceb
        IncFace: list[NDArray[np.float32]] = Facea
        Flip: bool = True

    Poly: list[NDArray[np.float32]] = IncFace.copy()

    for i in range(4):
        a: NDArray[np.float32] = RefFace[i]
        b: NDArray[np.float32] = RefFace[(i + 1) % 4]
        Edge: NDArray[np.float32] = b - a
        PlaneN: NDArray[np.float32] = normalize(np.cross(Edge, RefNormal))
        PlaneD: float = np.dot(PlaneN, a)

        Poly = clip_polygon(Poly, PlaneN, PlaneD)
        if not Poly:
            return []

    Contacts: list[Contact] = []

    for p in Poly:
        Depth: float = np.dot(RefNormal, RefCenter - p)
        if Depth >= 0:
            Contacts.append(
                Contact(
                    ObjectA=CubeA if not Flip else CubeB,
                    ObjectB=CubeB if not Flip else CubeA,
                    Point=p,
                    Normal=RefNormal if not Flip else -RefNormal,
                    Penetration=Depth,
                )
            )

    return Contacts


def generate_cs_contacts(Cube: P_Object, Sphere: P_Object) -> list[Contact]:
    Contacts: list[Contact] = []

    R: NDArray[np.float32] = Cube.RotationMatrix
    PLocal: NDArray[np.float32] = R.T @ (Sphere.Position - Cube.Position)  # type: ignore

    HalfExtent: NDArray[np.float32] = Cube.HalfExtent
    ClosestLocal: NDArray[np.float32] = np.clip(PLocal, -HalfExtent, HalfExtent)
    ClosestWorld: NDArray[np.float32] = Cube.Position + R @ ClosestLocal  # type: ignore

    Delta: NDArray[np.float32] = Sphere.Position - ClosestWorld
    Dist: float = np.linalg.norm(Delta)  # type: ignore

    if Dist >= Sphere.Side_Length:
        return Contacts

    if Dist > 1e-6:
        Normal: NDArray[np.float32] = Delta / Dist

    else:
        Normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    Penetration: float = Sphere.Side_Length - Dist
    Contacts.append(
        Contact(
            ObjectA=Cube,
            ObjectB=Sphere,
            Point=ClosestWorld,
            Normal=Normal,
            Penetration=Penetration,
        )
    )

    return Contacts


def solve_contacts(Contacts: list[Contact], Iterations: int = 8) -> None:
    for _ in range(Iterations):
        for C in Contacts:
            solve_contact(C)


def solve_contact(C: Contact) -> None:  # C here represents one contact
    A: P_Object = C.ObjectA
    B: P_Object = C.ObjectB

    InvMass_A: float = A.ReciprocalMass
    InvMass_B: float = B.ReciprocalMass

    Normal: NDArray[np.float32] = C.Normal
    Point: NDArray[np.float32] = C.Point

    Ra: NDArray[np.float32] = Point - A.Position
    Rb: NDArray[np.float32] = Point - B.Position

    Va: NDArray[np.float32] = A.Velocity + np.cross(A.AngularVelocity, Ra)  # type: ignore
    Vb: NDArray[np.float32] = B.Velocity + np.cross(B.AngularVelocity, Rb)  # type: ignore

    Vrel: NDArray[np.float32] = Vb - Va
    Vn: float = np.dot(Vrel, Normal)

    if Vn > 0.0:
        return

    e: float = min(A.Restitution, B.Restitution)
    if abs(Vn) < 0.5:
        e = 0.0

    InvMassSum: float = InvMass_A + InvMass_B

    if InvMass_A > 0.0:
        RaCrossN: NDArray[np.float32] = np.cross(Ra, Normal)
        InvMassSum += np.dot(Normal, np.cross(A.InvInertiaWorld @ RaCrossN, Ra))

    if InvMass_B > 0.0:
        RbCrossN: NDArray[np.float32] = np.cross(Rb, Normal)
        InvMassSum += np.dot(Normal, np.cross(B.InvInertiaWorld @ RbCrossN, Rb))

    if InvMassSum == 0.0:
        return

    Bias: float = 0.0
    if C.Penetration > SLOP and abs(Vn) < 0.5:
        Bias = BETA * (C.Penetration - SLOP) / DT

    j: float = -((1.0 + e) * Vn - Bias) / InvMassSum
    OldImpulse: float = C.AccmulatedNormalImpulse
    C.AccmulatedNormalImpulse = max(OldImpulse + j, 0.0)

    Delta_j: float = C.AccmulatedNormalImpulse - OldImpulse
    Impulse: NDArray[np.float32] = Delta_j * Normal

    if InvMass_A > 0.0:
        A.Velocity -= Impulse * InvMass_A
        A.AngularVelocity -= A.InvInertiaWorld @ np.cross(Ra, Impulse)

    if InvMass_B > 0.0:
        B.Velocity += Impulse * InvMass_B
        B.AngularVelocity += B.InvInertiaWorld @ np.cross(Rb, Impulse)

    A.Impulse -= Impulse
    B.Impulse += Impulse


def positional_correction(
    contacts: list[Contact], slop: float = 0.01, percent: float = 0.2
) -> None:
    for C in contacts:
        A: P_Object = C.ObjectA
        B: P_Object = C.ObjectB
        Normal: NDArray[np.float32] = C.Normal

        Depth: float = max(C.Penetration - slop, 0.0)

        if Depth <= 0.0:
            continue

        InvMassSum: float = A.ReciprocalMass + B.ReciprocalMass
        if InvMassSum == 0.0:
            continue

        Correction = percent * Depth / InvMassSum * Normal

        if A.ReciprocalMass > 0.0:
            A.Position -= Correction * A.ReciprocalMass
        if B.ReciprocalMass > 0.0:
            B.Position += Correction * B.ReciprocalMass


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

    BestDot: float = -1e9
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
