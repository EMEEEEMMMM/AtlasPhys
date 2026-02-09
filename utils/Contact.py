import numpy as np
from numpy.typing import NDArray

from utils.G_Object import P_Object
from utils.MathPhys_utils import normalize
from utils.Step import the_collision

SLOP: float = 0.02
BETA: float = 0.1
DT: float = 0.01


class Contact:
    __slots__ = ("Point", "Penetration", "Ra", "Rb", "NormalImpulse")

    def __init__(self, Point: NDArray[np.float32], Penetration: float) -> None:
        self.Point: NDArray[np.float32] = Point
        self.Penetration: float = Penetration
        self.Ra: NDArray[np.float32] | None = None
        self.Rb: NDArray[np.float32] | None = None
        self.NormalImpulse: float = 0.0


class ContactManifold:
    __slots__ = (
        "ObjectA",
        "ObjectB",
        "Normal",
        "Contacts",
        "AccmulatedTangentImpulse",
        "Key",
    )

    def __init__(
        self,
        ObjectA: P_Object,
        ObjectB: P_Object,
        Normal: NDArray[np.float32],
    ) -> None:
        self.ObjectA: P_Object = ObjectA
        self.ObjectB: P_Object = ObjectB
        self.Normal: NDArray[np.float32] = normalize(Normal)
        self.Contacts: list[Contact] = []
        self.AccmulatedTangentImpulse: float = 0.0
        self.Key: tuple[int, int] = (
            id(ObjectA),
            id(ObjectB),
        )


def generate_contacts(A: P_Object, B: P_Object) -> list[ContactManifold] | None:
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

            return generate_cc_contacts(A, B, Normal)

        case "Cube", "Sphere":
            return generate_cs_contacts(A, B)

        case "Sphere", "Cube":
            return generate_cs_contacts(B, A)

        case _:
            pass


def generate_pc_contacts(
    Plane: P_Object,
    Cube: P_Object,
) -> list[ContactManifold]:
    """
    Generate contacts of the plane and the cube

    Args:
        Plane (P_Object)
        Cube (P_Object)

    Returns:
        list[Contact]: a list containing all the contacts
    """
    Normal: NDArray[np.float32] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    Penetration: float = Plane.Position[1]

    Manifold: ContactManifold = ContactManifold(Plane, Cube, Normal)
    Deepest = None

    for vertex in Cube.XYZVertices:
        VertexWorld: NDArray[np.float32] = Cube.RotationMatrix @ vertex + Cube.Position
        Dist: float = np.dot(Normal, VertexWorld) - Penetration

        if Dist < 0.0:
            pen: float = -Dist
            if Deepest is None or pen > Deepest.Penetration:
                Deepest = Contact(VertexWorld, pen)

    if Deepest is not None:
        Manifold.Contacts.append(Deepest)
        return [Manifold]
    
    return []


def generate_ps_contacts(Plane: P_Object, Sphere: P_Object) -> list[ContactManifold]:
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
        return []

    Point: NDArray[np.float32] = Sphere.Position - Normal * Sphere.Side_Length  # type: ignore
    Penetration: float = Sphere.Side_Length - Dist

    Manifold: ContactManifold = ContactManifold(Plane, Sphere, Normal)
    Manifold.Contacts.append(Contact(Point, Penetration))

    return [Manifold]


def generate_ss_contacts(SphereA: P_Object, SphereB: P_Object) -> list[ContactManifold]:
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
        return []

    Normal: NDArray[np.float32] = d / (Dist + 1e-8)
    Point: NDArray[np.float32] = SphereA.Position + Normal * SphereA.Side_Length  # type: ignore

    Penetration: float = RadiusSum - Dist  # type: ignore

    Manifold: ContactManifold = ContactManifold(SphereA, SphereB, Normal)  # type: ignore
    Manifold.Contacts.append(Contact(Point, Penetration))

    return [Manifold]


def generate_cc_contacts(
    CubeA: P_Object, CubeB: P_Object, Normal: NDArray[np.float32]
) -> list[ContactManifold]:
    Normal = normalize(Normal)

    if np.dot(CubeB.Position - CubeA.Position, Normal) < 0:
        Normal = -Normal

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

    else:
        RefNormal: NDArray[np.float32] = Nb
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

        Poly = clip_polygon(Poly, PlaneN, PlaneD)
        if not Poly:
            return []

    Manifold: ContactManifold = ContactManifold(CubeA, CubeB, Normal)

    for p in Poly:
        Depth: float = np.dot(RefNormal, RefCenter - p)
        Manifold.Contacts.append(Contact(p, Depth))

    if not Manifold.Contacts:
        return []

    Manifold.Contacts.sort(key=lambda c: c.Penetration, reverse=True)
    Manifold.Contacts = [Manifold.Contacts[0]]
    return [Manifold]


def generate_cs_contacts(Cube: P_Object, Sphere: P_Object) -> list[ContactManifold]:
    R: NDArray[np.float32] = Cube.RotationMatrix
    PLocal: NDArray[np.float32] = R.T @ (Sphere.Position - Cube.Position)  # type: ignore

    HalfExtent: NDArray[np.float32] = Cube.HalfExtent
    ClosestLocal: NDArray[np.float32] = np.clip(PLocal, -HalfExtent, HalfExtent)
    ClosestWorld: NDArray[np.float32] = Cube.Position + R @ ClosestLocal  # type: ignore

    Delta: NDArray[np.float32] = Sphere.Position - ClosestWorld
    Dist: float = np.linalg.norm(Delta)  # type: ignore

    if Dist >= Sphere.Side_Length:
        return []

    if Dist > 1e-6:
        Normal: NDArray[np.float32] = Delta / Dist

    else:
        d: NDArray[np.float32] = Sphere.Position - Cube.Position
        if np.linalg.norm(d) > 1e-6:
            Normal = normalize(d)

        else:
            Normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    Penetration: float = Sphere.Side_Length - Dist

    Manifold: ContactManifold = ContactManifold(Cube, Sphere, Normal)
    Manifold.Contacts.append(Contact(ClosestWorld, Penetration))

    return [Manifold]


def solve_manifolds(Manifolds: list[ContactManifold], Iterations: int = 8) -> None:
    for M in Manifolds:
        A: P_Object = M.ObjectA
        B: P_Object = M.ObjectB
        Normal: NDArray[np.float32] = M.Normal

        for C in M.Contacts:
            C.Ra = C.Point - A.Position
            C.Rb = C.Point - B.Position

            if C.NormalImpulse > 0.0:
                Impulse: NDArray[np.float32] = C.NormalImpulse * Normal
                A.Velocity -= Impulse * A.ReciprocalMass
                A.AngularVelocity -= A.InvInertiaWorld @ np.cross(C.Ra, Impulse)

                B.Velocity += Impulse * B.ReciprocalMass
                B.AngularVelocity += B.InvInertiaWorld @ np.cross(C.Rb, Impulse)

    for _ in range(Iterations):
        for m in Manifolds:
            solve_manifold(m)


def solve_manifold(M: ContactManifold) -> None:  # C here represents one contact
    A: P_Object = M.ObjectA
    B: P_Object = M.ObjectB

    Normal: NDArray[np.float32] = M.Normal
    Contacts: list[Contact] = M.Contacts

    if A.Shape == "Cube" and B.Shape == "Cube":
        C = M.Contacts[0]
        Ra = C.Ra
        Rb = C.Rb

        Va = A.Velocity + np.cross(A.AngularVelocity, Ra)
        Vb = B.Velocity + np.cross(B.AngularVelocity, Rb)

        Vn = np.dot(Vb - Va, Normal)

        with open("log.txt", "a") as f:
            print("A.y=", A.Position[1], "B.y=", B.Position[1], file=f)
            print("Normal=", Normal, file=f)
            print("Vn=", Vn, file=f)

    if len(Contacts) == 0:
        return

    if len(Contacts) == 1:
        C: Contact = Contacts[0]
        Ra: NDArray[np.float32] = C.Ra  # type: ignore
        Rb: NDArray[np.float32] = C.Rb  # type: ignore

        Va: NDArray[np.float32] = A.Velocity + np.cross(A.AngularVelocity, Ra)  # type: ignore
        Vb: NDArray[np.float32] = B.Velocity + np.cross(B.AngularVelocity, Rb)  # type: ignore
        Vn: np.float32 = np.dot(Vb - Va, Normal)  # type: ignore

        if Vn > 0 and C.Penetration <= 0:
            return

        k: float = (
            A.ReciprocalMass
            + B.ReciprocalMass
            + np.dot(
                np.cross(A.InvInertiaWorld @ np.cross(Ra, Normal), Ra)
                + np.cross(B.InvInertiaWorld @ np.cross(Rb, Normal), Rb),
                Normal,
            )
        )

        if k == 0:
            return

        Bias: float = 0.0
        if C.Penetration > SLOP:
            Bias = BETA * (C.Penetration - SLOP) / DT

        j: float = -(Vn + Bias) / k  # type: ignore
        Old: float = C.NormalImpulse
        C.NormalImpulse = max(Old + j, 0.0)
        Delta_j: float = C.NormalImpulse - Old

        Impulse: NDArray[np.float32] = Delta_j * Normal
        apply_impulse_point(A, B, Impulse, Ra, Rb)

        return

    K: NDArray[np.float32] = np.zeros((2, 2), dtype=np.float32)
    Vn: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
    Bias: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
    MaxPenetration: float = max(c.Penetration for c in M.Contacts)

    for i in range(2):
        C_i: Contact = Contacts[i]
        Ra_i: NDArray[np.float32] = C_i.Ra  # type: ignore
        Rb_i: NDArray[np.float32] = C_i.Rb  # type: ignore

        V_i: NDArray[np.float32] = (
            B.Velocity + np.cross(B.AngularVelocity, Rb_i) - A.Velocity - np.cross(A.AngularVelocity, Ra_i)  # type: ignore
        )
        Vn[i] = np.dot(V_i, Normal)
        
        if C_i.Penetration > SLOP:
            Bias[i] = BETA * (C_i.Penetration - SLOP) / DT

        for j in range(2):
            C_j: Contact = Contacts[j]
            Ra_j: NDArray[np.float32] = C_j.Ra  # type: ignore
            Rb_j: NDArray[np.float32] = C_j.Rb  # type: ignore

            K[i, j] = (  # type: ignore
                A.ReciprocalMass
                + B.ReciprocalMass
                + np.dot(
                    np.cross(A.InvInertiaWorld @ np.cross(Ra_i, Normal), Ra_j)
                    + np.cross(B.InvInertiaWorld @ np.cross(Rb_i, Normal), Rb_j),
                    Normal,
                )
            )

    if Vn[0] > 0.0 and Vn[1] > 0.0 and MaxPenetration <= 0.0:
        return

    Lambda_Old = np.array(
        [Contacts[0].NormalImpulse, Contacts[1].NormalImpulse], dtype=np.float32
    )

    b = Vn + Bias + K @ Lambda_Old

    try:
        DeltaLambda = np.linalg.solve(K, -b)
    except np.linalg.LinAlgError:
        for i in range(2):
            C_i = Contacts[i]
            if K[i, i] > 1e-6:
                j_i = -(Vn[i] + Bias[i]) / K[i, i]
                Old = C_i.NormalImpulse
                C_i.NormalImpulse = max(Old + j_i, 0.0)
                Delta_j = C_i.NormalImpulse - Old
                apply_impulse_point(A, B, Delta_j * Normal, C_i.Ra, C_i.Rb)  # type: ignore
        return

    Lambda_New = np.maximum(Lambda_Old + DeltaLambda, 0.0)

    for i in range(2):
        Delta_j = Lambda_New[i] - Lambda_Old[i]
        Contacts[i].NormalImpulse = Lambda_New[i]
        with open("log_solve.txt", "a") as f:
            print("Normal=", Normal, file=f)
            print("Impulse=", Delta_j * Normal, file=f)
            print("A before Vy=", A.Velocity[1], file=f)
            print("B before Vy=", B.Velocity[1], file=f)
        apply_impulse_point(A, B, Delta_j * Normal, Contacts[i].Ra, Contacts[i].Rb)  # type: ignore


def positional_correction_manifold(
    M: ContactManifold, slop: float = 0.005, percent: float = 0.9
) -> None:
    A: P_Object = M.ObjectA
    B: P_Object = M.ObjectB
    if A.Shape == "Cube" and B.Shape == "Cube":
        return
    
    Normal: NDArray[np.float32] = M.Normal

    Penetration: float = max(C.Penetration for C in M.Contacts)
    Depth: float = max(Penetration - slop, 0.0)

    if Depth <= 0.0:
        return

    InvMassSum: float = A.ReciprocalMass + B.ReciprocalMass
    if InvMassSum == 0.0:
        return

    Correction = percent * Depth / InvMassSum * Normal

    if A.ReciprocalMass > 0.0:
        A.Position -= Correction * A.ReciprocalMass
    if B.ReciprocalMass > 0.0:
        B.Position += Correction * B.ReciprocalMass


def apply_impulse_point(
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

    A.Impulse -= J
    B.Impulse += J

    if A.Shape == "Cube" and B.Shape == "Cube":
        with open("log_impulse.txt", "a") as f:
            print("A after Vy=", A.Velocity[1], file=f)
            print("B after Vy=", B.Velocity[1], file=f)


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
