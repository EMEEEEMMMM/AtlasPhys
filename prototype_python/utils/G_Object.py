import numpy as np
from typing import Any
from numpy.typing import NDArray

from utils import MathPhys_utils


class P_Object:
    """
    This includes all the properties of all the substance in this phyics engine.
    """

    def __init__(
        self,
        Shape: str,
        Side_Length: float,
        X_Coordinate: float,
        Y_Coordinate: float,
        Z_Coordinate: float,
        R_v: float,
        G_v: float,
        B_v: float,
        A_v: float,
        Mass: float,
        Restitution: float,
        GL_Type: int,
        Vertices: NDArray[np.float32],
        Indices: NDArray[np.uint32],
        Vao: int,
        Vbo: int,
        Ebo: int,
    ) -> None:
        self.Shape: str = Shape
        self.Side_Length: float = Side_Length
        self.Position: NDArray[np.float32] = np.array(
            [
                X_Coordinate,
                Y_Coordinate,
                Z_Coordinate,
            ],
            dtype=np.float32,
        )
        self.R_v: float = R_v
        self.G_v: float = G_v
        self.B_v: float = B_v
        self.A_v: float = A_v
        self.GL_Type: int = GL_Type
        self.Vertices: NDArray[np.float32] = Vertices
        self.Indices: NDArray[np.uint32] = Indices
        self.VAO: int = Vao
        self.VBO: int = Vbo
        self.EBO: int = Ebo

        self.Len_Indices: int = len(self.Indices)
        self.Velocity: NDArray[np.float32] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.Acceleration: NDArray[np.float32] = np.array(
            [0.0, 0.0, 0.0], dtype=np.float32
        )
        self.Scale: NDArray[np.float32] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.Rotation: NDArray[np.float32] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.RotationMatrix: NDArray[np.float32] = MathPhys_utils.euler_to_matrix(
            self.Rotation
        )
        self.AngularVelocity: NDArray[np.float32] = np.array(
            [0.0, 0.0, 0.0], dtype=np.float32
        )
        self.MASS: float = Mass
        self.Restitution: float = Restitution

        self.Collidable = True
        self.ReciprocalMass: float = 1 / self.MASS if self.MASS != float("inf") else 0.0
        self.XYZVertices: NDArray[np.float32] = np.ascontiguousarray(
            Vertices.reshape(-1, 7)[:, :3]
        )
        self.Impulse: NDArray[np.float32] = np.zeros(3, dtype=np.float32)
        self.Fnet_Impulse: NDArray[np.float32] = np.zeros(
            3, dtype=np.float32
        )
        self.Fnet_MA: NDArray[np.float32] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.Impulse_Arrow: Impulse_Arrow | None = None
        self.MA_Arrow: MA_Arrow | None = None

        if Shape == "Sphere":
            self.InertiaBody: NDArray[np.float32] = np.float32(
                (2 / 5) * self.MASS * (Side_Length**2)
            ) * np.eye(3, dtype=np.float32)

            self.InvInertiaBody: NDArray[np.float32] = np.linalg.inv(self.InertiaBody)
            self.InvInertiaWorld: NDArray[np.float32] = (
                self.RotationMatrix
                @ self.InvInertiaBody
                @ np.transpose(self.RotationMatrix)
            ).astype(np.float32)

        if Shape == "Cube":
            self.InertiaBody = np.float32(
                (1 / 12) * self.MASS * (Side_Length**2)
            ) * np.diag(np.array([2.0, 2.0, 2.0], dtype=np.float32))

            self.InvInertiaBody: NDArray[np.float32] = np.linalg.inv(self.InertiaBody)
            self.InvInertiaWorld: NDArray[np.float32] = (
                self.RotationMatrix
                @ self.InvInertiaBody
                @ np.transpose(self.RotationMatrix)
            ).astype(np.float32)
            self.HalfExtent: NDArray[np.float32] = np.array(
                [self.Side_Length / 2, self.Side_Length / 2, self.Side_Length / 2],
                dtype=np.float32,
            )

        self._get_aabb()

    def get_model_matrix(self) -> NDArray[np.float32]:
        ScaleMatrix: NDArray[np.float32] = MathPhys_utils.scale_matrix(*self.Scale)
        RotationMatrix: NDArray[np.float32] = MathPhys_utils.rotation_matrix(
            *self.Rotation
        )
        TranslationMatrix: NDArray[np.float32] = MathPhys_utils.translation_matrix(
            *self.Position
        )

        FinalMatrix: NDArray[np.float32] = TranslationMatrix @ RotationMatrix @ ScaleMatrix  # type: ignore

        return FinalMatrix

    def _get_aabb(self) -> None:
        MaxXYZ: NDArray[np.uint8] = np.argmax(self.XYZVertices, axis=0)
        self.X_MAX: float = self.XYZVertices[MaxXYZ[0], 0]
        self.Y_MAX: float = self.XYZVertices[MaxXYZ[1], 1]
        self.Z_MAX: float = self.XYZVertices[MaxXYZ[2], 2]

        MinXYZ: NDArray[np.uint8] = np.argmin(self.XYZVertices, axis=0)
        self.X_MIN: float = self.XYZVertices[MinXYZ[0], 0]
        self.Y_MIN: float = self.XYZVertices[MinXYZ[1], 1]
        self.Z_MIN: float = self.XYZVertices[MinXYZ[2], 2]

    def get_rotation_matrix(self) -> NDArray[np.float32]:
        return MathPhys_utils.rotation_matrix(*self.Rotation)


class Coordinate_Axis(P_Object):
    """
    The special case for the CoordinateAxis since it should not be moving at any time
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.Collidable = False

    def update_position(self, DeltaTime: float) -> None:
        pass


class Plane(P_Object):
    """
    The special case for the Plane since it should not be moving at any time :)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.MASS = float("inf")
        self.ReciprocalMass = 0.0
        self.InvInertiaBody = np.zeros((3, 3), dtype=np.float32)
        self.InvInertiaWorld = np.zeros((3, 3), dtype=np.float32)
        self.Collidable = True

    def update_position(self, DeltaTime: float) -> None:
        pass


class MA_Arrow:  # Arrows calculated from object's mass and acceleration which always appear  Fnet = mass * acceleration
    def __init__(self, Vao: int, Vbo: int) -> None:
        self.Vao: int = Vao
        self.Vbo: int = Vbo
        self.VertexCount: int = 0


class Impulse_Arrow:  # Arrows calculated from impulse which only appear when collision happens  Fnet = Impulse / deltatime
    def __init__(self, Vao: int, Vbo: int) -> None:
        self.Vao: int = Vao
        self.Vbo: int = Vbo
        self.VertexCount: int = 0
