import numpy as np
from typing import TypedDict
from numpy.typing import NDArray
from utils import MathPhys_utils


class Object_DataType(TypedDict):
    Shape: str
    Side_Length: float
    X_Coordinate: float
    Y_Coordinate: float
    Z_Coordinate: float
    R_v: float
    G_v: float
    B_v: float
    A_v: float
    GL_Type: int
    Vertices: NDArray[np.float32]
    Indices: NDArray[np.uint32]
    Vao: int
    Vbo: int
    Ebo: int


class P_Object:
    """
    This includes all the properties of all the substance in this phyics engine.
    """

    def __init__(
        self,
        Object_Data: Object_DataType,
    ) -> None:
        self.Shape: str = Object_Data["Shape"]
        self.Side_Length: float = Object_Data["Side_Length"]
        self.Postion: NDArray[np.float32] = np.array(
            [
                Object_Data["X_Coordinate"],
                Object_Data["Y_Coordinate"],
                Object_Data["Z_Coordinate"],
            ],
            dtype=np.float32,
        )
        self.R_v: float = Object_Data["R_v"]
        self.G_v: float = Object_Data["G_v"]
        self.B_v: float = Object_Data["B_v"]
        self.A_v: float = Object_Data["A_v"]
        self.GL_Type: int = Object_Data["GL_Type"]
        self.Vertices: NDArray[np.float32] = Object_Data["Vertices"]
        self.Indices: NDArray[np.uint32] = Object_Data["Indices"]
        self.VAO: int = Object_Data["Vao"]
        self.VBO: int = Object_Data["Vbo"]
        self.EBO: int = Object_Data["Ebo"]

        self.Len_Indices: int = len(self.Indices)
        self.Velocity: NDArray[np.float32] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.Acceleration: NDArray[np.float32] = np.array(
            [0.0, 0.0, 0.0], dtype=np.float32
        )
        self.Scale: NDArray[np.float32] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.Rotation: NDArray[np.float32] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.AngularVelocity: NDArray[np.float32] = np.array(
            [0.0, 0.0, 0.0], dtype=np.float32
        )

        self.MASS: np.uint32

    def update_position(self, DeltaTime: float) -> None:
        """
        Update the position,velocity,acceleration of the object every deltatime

        Args:
            DeltaTime (float): delta time
        """
        self.update_gravity()

        Displacement_X: float = MathPhys_utils.displacement_Uacceleration(
            self.Velocity[0], self.Acceleration[0], DeltaTime
        )
        Displacement_Y: float = MathPhys_utils.displacement_Uacceleration(
            self.Velocity[1], self.Acceleration[1], DeltaTime
        )
        Displacement_Z: float = MathPhys_utils.displacement_Uacceleration(
            self.Velocity[2], self.Acceleration[2], DeltaTime
        )

        self.Postion[0] += Displacement_X
        self.Postion[1] += Displacement_Y
        self.Postion[2] += Displacement_Z

        self.Rotation[0] += self.AngularVelocity[0] * DeltaTime
        self.Rotation[1] += self.AngularVelocity[1] * DeltaTime
        self.Rotation[2] += self.AngularVelocity[2] * DeltaTime

        self.Velocity[0] += self.Acceleration[0] * DeltaTime
        self.Velocity[1] += self.Acceleration[1] * DeltaTime
        self.Velocity[2] += self.Acceleration[2] * DeltaTime

    def get_model_matrix(self) -> NDArray[np.float32]:
        ScaleMatrix: NDArray[np.float32] = MathPhys_utils.scale_matrix(*self.Scale)
        RotationMatrix: NDArray[np.float32] = MathPhys_utils.rotation_matrix(
            *self.Rotation
        )
        TranslationMatrix: NDArray[np.float32] = MathPhys_utils.translation_matrix(
            *self.Postion
        )

        FinalMatrix: NDArray[np.float32] = TranslationMatrix @ RotationMatrix @ ScaleMatrix  # type: ignore

        return FinalMatrix

    def update_gravity(self) -> None:
        self.Acceleration[1] = MathPhys_utils.GRAVITY

    def detect_collision(self) -> bool: ...


class Coordinate_Axis(P_Object):
    """
    The special case for the CoordinateAxis since it should not be moving at any time
    """

    def __init__(self, Object_Data: Object_DataType) -> None:
        super().__init__(Object_Data)

    def update_position(self, DeltaTime: float) -> None:
        pass

    def update_gravity(self) -> None:
        pass


class Plane(P_Object):
    """
    The special case for the Plane since it should not be moving at any time :)
    """

    def __init__(self, Object_Data: Object_DataType) -> None:
        super().__init__(Object_Data)

    def update_position(self, DeltaTime: float) -> None:
        pass

    def update_gravity(self) -> None:
        pass
