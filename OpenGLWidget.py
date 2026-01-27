from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
from OpenGL.GL import *  # type: ignore
from PyQt6.QtCore import QElapsedTimer, QModelIndex, QPointF, Qt, QTimer
from PyQt6.QtGui import QSurfaceFormat, QMouseEvent, QKeyEvent, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import random
import time
import threading

from utils import MathPhys_utils, Opengl_utils, Generate_Objects
from utils.G_Object import P_Object, Coordinate_Axis, Plane
from utils import Step


class Simulator(QOpenGLWidget):

    def __init__(self, window_ui: Any, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.window_self: Any = window_ui

        # Camera
        self.CameraPos: NDArray[np.float32] = np.array(
            [0.0, 0.0, 3.0], dtype=np.float32
        )
        self.CameraFront: NDArray[np.float32] = np.array(
            [0.0, 0.0, -1.0], dtype=np.float32
        )
        self.CameraUP: NDArray[np.float32] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.Scale_K: NDArray[np.float32] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.IS_PERSPECTIVE: bool = True
        self.COORDINATE_AXIS: bool = False
        self.DemoLoaded: bool = False
        self.PLANE: bool = False
        self.START_OR_STOP: bool = True
        self.VIEW: NDArray[np.float32] = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])
        self.Yaw: float = -90.0
        self.Pitch: float = 0.0

        self.RightDown: bool = False
        self.LastPos: QPointF = QPointF()
        self.Graphics: list[P_Object] = []

        self.TIMER = QElapsedTimer()
        self.Time_Started: bool = False
        self.Animation_Timer = QTimer(self)
        self.Animation_Timer.timeout.connect(self.update)
        self.Animation_Timer.start(7)

        self.LastTime: float = self.get_time()

        self.Accmulator: float = 0.0
        self.PhysicsStep: float = 0.01

    def initializeGL(self) -> None:
        self.fmt = QSurfaceFormat()
        self.fmt.setVersion(3, 3)
        self.fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.fmt.setDepthBufferSize(24)
        self.fmt.setStencilBufferSize(8)
        self.setFormat(self.fmt)
        self.setAutoFillBackground(False)

        glEnable(GL_DEPTH_TEST)
        # glEnable(GL_CULL_FACE)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        print("当前渲染器:", glGetString(GL_RENDERER).decode())  # type: ignore
        # Enable depth testing to achieve occlusion relationships
        self.shader_program = self.compile_shaders()
        glDepthFunc(GL_LEQUAL)
        # Set the depth test function (GL_LEQUAL is just one of the options)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # type: ignore
        # self.draw_plane()
        # self.draw_sun()

    def paintGL(self) -> None:
        self.makeCurrent()
        w: int
        h: int
        w, h = self.width(), self.height()

        CurrentTime: float = self.get_time()
        DeltaTime: float = CurrentTime - self.LastTime
        self.LastTime = CurrentTime

        if DeltaTime > 0.05:
            DeltaTime = 0.05

        if self.IS_PERSPECTIVE:
            # Perspective Projection (glFrustum)
            if w > h:
                left: np.float32 = self.VIEW[0] * w / h
                right: np.float32 = self.VIEW[1] * w / h
                bottom: np.float32 = self.VIEW[2]
                top: np.float32 = self.VIEW[3]
            else:
                left: np.float32 = self.VIEW[0]
                right: np.float32 = self.VIEW[1]
                bottom: np.float32 = self.VIEW[2] * h / w
                top: np.float32 = self.VIEW[3] * h / w
            projection: NDArray[np.float32] = Opengl_utils.perspective_projection(
                left, right, bottom, top, self.VIEW[4], self.VIEW[5]
            )

        else:
            # Orthogonal Projection (glOrtho)
            if w > h:
                left = self.VIEW[0] * w / h
                right = self.VIEW[1] * w / h
                bottom = self.VIEW[2]
                top = self.VIEW[3]
            else:
                left = self.VIEW[0]
                right = self.VIEW[1]
                bottom = self.VIEW[2] * h / w
                top = self.VIEW[3] * h / w
            projection = Opengl_utils.ortho_projection(
                left, right, bottom, top, self.VIEW[4], self.VIEW[5]
            )

        # Radius: float = 10.0
        # CamX: np.float32 = np.sin(self.get_time()) * Radius
        # CamZ: np.float32 = np.cos(self.get_time()) * Radius

        # Vector1 = np.array([CamX, 0.0, CamZ], dtype=np.float32)
        # Vector2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Vector3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        ViewMatrix: NDArray[np.float32] = Opengl_utils.lookat(
            self.CameraPos, self.CameraPos + self.CameraFront, self.CameraUP
        )

        glUseProgram(self.shader_program)

        ProjLocation = glGetUniformLocation(self.shader_program, "projection")
        ViewLocation = glGetUniformLocation(self.shader_program, "view")
        ModelLocation = glGetUniformLocation(self.shader_program, "model")

        glUniformMatrix4fv(ProjLocation, 1, GL_FALSE, projection.T.flatten())
        glUniformMatrix4fv(ViewLocation, 1, GL_FALSE, ViewMatrix.T.flatten())

        LightColor = glGetUniformLocation(self.shader_program, "lightColor")
        glUniform4f(LightColor, 1.0, 1.0, 1.0, 1.0)

        if self.START_OR_STOP:
            self.Accmulator += DeltaTime

            while self.Accmulator >= self.PhysicsStep:
                Positions: NDArray[np.float32]
                Velocities: NDArray[np.float32]
                DynamicObjects: list[P_Object]
                Positions, Velocities, DynamicObjects = Step.extract_data(self.Graphics)

                Step.integrator(Positions, Velocities, self.PhysicsStep)

                Step.update_data(Positions, Velocities, DynamicObjects)

                for obj in DynamicObjects:
                    if obj.Shape == "Cube":
                        Step.solve_ground_collision_cube(obj)

                    elif obj.Shape == "Sphere":
                        Step.solve_ground_collision_sphere(obj)
                        
                for idx, obj in enumerate(DynamicObjects):
                    for i in range(idx + 1, len(DynamicObjects)):
                        Step.the_collision(obj, DynamicObjects[i])

                self.Accmulator -= self.PhysicsStep

        for obj in self.Graphics:
            ModelMatrix: NDArray[np.float32] = obj.get_model_matrix()
            ScaleMatrix: NDArray[np.float32] = Opengl_utils.scalef(self.Scale_K)
            ModelMatrix = ScaleMatrix @ ModelMatrix
            glUniformMatrix4fv(ModelLocation, 1, GL_FALSE, ModelMatrix.T.flatten())

            glBindVertexArray(obj.VAO)
            glDrawElements(
                obj.GL_Type, obj.Len_Indices, GL_UNSIGNED_INT, ctypes.c_void_p(0)
            )
            glBindVertexArray(0)

        # Clear the buffer and send the instructions to the hardware for immediate execution
        glFlush()

    def mousePressEvent(self, a0: QMouseEvent) -> None:
        self.setFocus()
        if a0.button() == Qt.MouseButton.RightButton:
            self.RightDown = True
            self.LastPos = a0.position()

    def mouseMoveEvent(self, a0: QMouseEvent) -> None:
        if not self.RightDown:
            return

        Current_Position: QPointF = a0.position()
        DX: float = Current_Position.x() - self.LastPos.x()
        DY: float = self.LastPos.y() - Current_Position.y()

        Sensitivity: float = 0.05
        self.Yaw += DX * Sensitivity
        self.Pitch += DY * Sensitivity

        self.Pitch = np.clip(self.Pitch, -89.0, 89.0)

        Front: NDArray[np.float32] = np.array(
            [
                np.cos(np.radians(self.Yaw)) * np.cos(np.radians(self.Pitch)),
                np.sin(np.radians(self.Pitch)),
                np.sin(np.radians(self.Yaw)) * np.cos(np.radians(self.Pitch)),
            ],
            dtype=np.float32,
        )
        self.CameraFront = MathPhys_utils.normalize(Front)

        self.LastPos = Current_Position

        self.update()

    def mouseReleaseEvent(self, a0: QMouseEvent) -> None:
        if a0.button() == Qt.MouseButton.RightButton:
            self.RightDown = False

    def keyPressEvent(self, a0: QKeyEvent) -> None:
        CameraSpeed: float = 10.0 * 1 / 120

        match a0.key():
            case Qt.Key.Key_W:
                self.CameraPos += CameraSpeed * self.CameraFront

            case Qt.Key.Key_S:
                self.CameraPos -= CameraSpeed * self.CameraFront

            case Qt.Key.Key_A:
                self.CameraPos -= (
                    MathPhys_utils.normalize(np.cross(self.CameraFront, self.CameraUP))
                    * CameraSpeed
                )

            case Qt.Key.Key_D:
                self.CameraPos += (
                    MathPhys_utils.normalize(np.cross(self.CameraFront, self.CameraUP))
                    * CameraSpeed
                )

            case Qt.Key.Key_Q:
                self.CameraPos += CameraSpeed * self.CameraUP

            case Qt.Key.Key_E:
                self.CameraPos -= CameraSpeed * self.CameraUP

            case _:
                pass

    def wheelEvent(self, a0: QWheelEvent) -> None:
        delta: int = a0.angleDelta().y()
        if delta > 0:
            self.Scale_K *= 1.05
        else:
            self.Scale_K *= 0.95

        self.update()

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)
        self.update()

    def compile_shaders(self):
        """
        To use the shader

        Returns:
            shaders program
        """
        # Vertex_shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(
            vertex_shader, Opengl_utils.load_shader("ShaderProgram/vertex_shader.vert")
        )
        glCompileShader(vertex_shader)

        # Vertex_shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(
            fragment_shader,
            Opengl_utils.load_shader("ShaderProgram/fragment_shader.frag"),
        )
        glCompileShader(fragment_shader)

        # shader_program
        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return shader_program

    def draw_coordinates(self) -> None:
        """
        To generate a coordinate axis on the screen
        """
        self.makeCurrent()

        # fmt: off
        Vertices: NDArray[np.float32] = np.array([
            # X-axis
            -100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            
            # Y-axis
            0.0, -100.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 100.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            
            # Z-axis
            0.0, 0.0, -100.0, 0.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 100.0, 0.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)

        Indices: NDArray[np.uint32] = np.array([
            0, 1,
            2, 3,
            4, 5
        ], dtype=np.uint32)
        # fmt: on

        Vao, Vbo, Ebo = Opengl_utils.analysis_data(self, Vertices, Indices)
        ObjectCompleteData: dict[str, Any] = {
            "Shape": "CoordinateAxis",
            "Side_Length": 0.0,
            "Side_Length": 0.0,
            "X_Coordinate": 0.0,
            "Y_Coordinate": 0.0,
            "Z_Coordinate": 0.0,
            "R_v": 1.0,
            "G_v": 1.0,
            "B_v": 1.0,
            "A_v": 1.0,
            "Mass": float("inf"),
            "Restitution": 0.0,
            "GL_Type": GL_LINES,
            "Vertices": Vertices,
            "Indices": Indices,
            "Vao": Vao,
            "Vbo": Vbo,
            "Ebo": Ebo,
        }
        self.CoordinateObj: P_Object = Coordinate_Axis(**ObjectCompleteData)
        self.Graphics.append(self.CoordinateObj)

        index: int = len(self.Graphics)

        self.window_self.ObjectList.beginInsertRows(QModelIndex(), index, index)
        self.window_self.ObjectList.endInsertRows()

        self.COORDINATE_AXIS = True  # type: ignore

        self.update()

    def draw_plane(self) -> None:
        """
        To generate a plane on the screen
        """
        self.makeCurrent()

        # fmt: off
        Vertices: NDArray[np.float32] = np.array(
            [
                -500, 0.0, 500, 0.5, 0.5, 0.5, 1.0,
                500, 0.0, 500, 0.5, 0.5, 0.5, 1.0,
                500, 0.0, -500, 0.5, 0.5, 0.5, 1.0,
                -500, 0.0, -500, 0.5, 0.5, 0.5, 1.0,

                -500, -2.0, 500, 0.5, 0.5, 0.5, 1.0,
                500, -2.0, 500, 0.5, 0.5, 0.5, 1.0,
                500, -2.0, -500, 0.5, 0.5, 0.5, 1.0,
                -500, -2.0, -500, 0.5, 0.5, 0.5, 1.0,
            ],
            dtype=np.float32,
        )

        Indices: NDArray[np.uint32] = np.array([
            0, 1, 2, 2, 3, 0,       
            4, 5, 6, 6, 7, 4,       
            0, 1, 5, 5, 4, 0,       
            2, 3, 7, 7, 6, 2,       
            1, 2, 6, 6, 5, 1,       
            3, 0, 4, 4, 7, 3
        ],dtype=np.uint32)

        # fmt: on

        Vao, Vbo, Ebo = Opengl_utils.analysis_data(self, Vertices, Indices)

        ObjectCData: dict[str, Any] = {
            "Shape": "Plane",
            "Side_Length": 1000,
            "X_Coordinate": 0.0,
            "Y_Coordinate": -1.0,
            "Z_Coordinate": 0.0,
            "R_v": 0.5,
            "G_v": 0.5,
            "B_v": 0.5,
            "A_v": 1.0,
            "Mass": float("inf"),
            "Restitution": 1.0,
            "GL_Type": GL_TRIANGLES,
            "Vertices": Vertices,
            "Indices": Indices,
            "Vao": Vao,
            "Vbo": Vbo,
            "Ebo": Ebo,
        }

        self.PlaneObj: P_Object = Plane(**ObjectCData)

        self.Graphics.append(self.PlaneObj)
        index: int = len(self.Graphics)

        self.window_self.ObjectList.beginInsertRows(QModelIndex(), index, index)
        self.window_self.ObjectList.endInsertRows()

        self.PLANE = True  # type: ignore

        self.update()

    # def draw_sun(self) -> None:
    #     """
    #     Create a light for the entire environment

    #     TODO: A delete or add muti-light function
    #     """
    #     self.makeCurrent()
    #     Vertices, Indices = Generate_Objects.generate_sphere(
    #         5.0, 0.0, 10.0, 0.0, 100, 100
    #     )

    #     # Vao
    #     self.SunVao: int = glGenVertexArrays(1)
    #     glBindVertexArray(self.SunVao)  # type: ignore

    #     # Vbo
    #     self.SunVbo: int = glGenBuffers(1)
    #     glBindBuffer(GL_ARRAY_BUFFER, self.SunVbo)  # type: ignore
    #     glBufferData(GL_ARRAY_BUFFER, Vertices.nbytes, Vertices, GL_STATIC_DRAW)

    #     # Ebo
    #     self.SunEbo: int = glGenBuffers(1)
    #     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.SunEbo)  # type: ignore
    #     glBufferData(GL_ELEMENT_ARRAY_BUFFER, Indices.nbytes, Indices, GL_STATIC_DRAW)

    #     # Vertices
    #     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(0))
    #     glEnableVertexAttribArray(0)

    #     # Colors
    #     glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4))
    #     glEnableVertexAttribArray(1)

    #     glBindVertexArray(0)
    #     glBindBuffer(GL_ARRAY_BUFFER, 0)
    #     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    #     self.Graphics.append(
    #         (self.SunVao, self.SunVbo, self.SunEbo, len(Indices), GL_TRIANGLES)  # type: ignore
    #     )

    #     self.update()

    def delete_single_object(self, vao: int, vbo: int, ebo: int) -> None:
        """
        To release the buffer of the object's vao, vbo, ebo

        Args:
            vao
            vbo
            ebo
        """
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
        glDeleteBuffers(1, [ebo])

    def get_time(self) -> float:
        """
        Get the Deltatime for every loop

        Returns:
            float: the deltatime
        """
        if not self.Time_Started:
            self.TIMER.start()
            self.Time_Started = True

        return self.TIMER.elapsed() / 1000.0

    def start_render(self) -> None:
        """
        Start the entire loop
        """
        self.LastTime = self.get_time()

    def load_demo(self) -> None:
        """
        Load the demo
        """
        self.window_self.AddOrDeletePlane.clicked.emit()
        self.DemoLoaded = True
        self.DemoThread = threading.Thread(target=self._demo_loop, daemon=True)
        self.DemoThread.start()

    def unload_demo(self) -> None:
        """
        Unload the demo
        """
        self.DemoLoaded = False
        self.DemoThread.join()
        self.window_self.ObjectList.clear_all()
        self.COORDINATE_AXIS = False  # type: ignore
        self.PLANE = False  # type: ignore

        self.update()

    def _demo_loop(self) -> None:
        for _ in range(10):
            if self.DemoLoaded and self.START_OR_STOP:
                self.window_self.AddCube.clicked.emit()
                self.window_self.AddSphere.clicked.emit()
                time.sleep(1.0)

    def add_demo_cube(self) -> None:
        self.makeCurrent()

        R_v: float = random.randint(0, 255) / 255.0
        G_v: float = random.randint(0, 255) / 255.0
        B_v: float = random.randint(0, 255) / 255.0

        if R_v == 0.0 and G_v == 0.0 and B_v == 0.0:
            R_v: float = random.randint(0, 255) / 255.0
            G_v: float = random.randint(0, 255) / 255.0
            B_v: float = random.randint(0, 255) / 255.0

        IData: dict[str, Any] = {
            "Side_Length": 2.0,
            "X_Coordinate": random.randint(-30, 30),
            "Y_Coordinate": random.randint(10, 15),
            "Z_Coordinate": random.randint(-30, 30),
            "R_v": R_v,
            "G_v": G_v,
            "B_v": B_v,
            "A_v": 1.0,
        }

        VIDATA = Generate_Objects.add_cube(**IData)

        Vao, Vbo, Ebo = Opengl_utils.analysis_data(
            self, VIDATA["Vertices"], VIDATA["Indices"]
        )

        CData: dict[str, Any] = (
            {"Shape": "Cube"}
            | IData
            | {"Mass": 3.0, "Restitution": 0.3}
            | VIDATA
            | {"Vao": Vao, "Vbo": Vbo, "Ebo": Ebo}
        )

        OneOfTheCube: P_Object = P_Object(**CData)
        self.Graphics.append(OneOfTheCube)

        index: int = len(self.Graphics)

        self.window_self.ObjectList.beginInsertRows(QModelIndex(), index, index)
        self.window_self.ObjectList.endInsertRows()

        self.update()

    def add_demo_sphere(self) -> None:
        self.makeCurrent()

        R_v: float = random.randint(0, 255) / 255.0
        G_v: float = random.randint(0, 255) / 255.0
        B_v: float = random.randint(0, 255) / 255.0

        if R_v == 0.0 and G_v == 0.0 and B_v == 0.0:
            R_v: float = random.randint(0, 255) / 255.0
            G_v: float = random.randint(0, 255) / 255.0
            B_v: float = random.randint(0, 255) / 255.0

        IData: dict[str, Any] = {
            "Side_Length": 2.0,
            "X_Coordinate": random.randint(-30, 30),
            "Y_Coordinate": random.randint(10, 15),
            "Z_Coordinate": random.randint(-30, 30),
            "R_v": R_v,
            "G_v": G_v,
            "B_v": B_v,
            "A_v": 1.0,
        }

        VIDATA = Generate_Objects.add_sphere(**IData)

        Vao, Vbo, Ebo = Opengl_utils.analysis_data(
            self, VIDATA["Vertices"], VIDATA["Indices"]
        )

        CData: dict[str, Any] = (
            {"Shape": "Sphere"}
            | IData
            | {"Mass": 3.0, "Restitution": 0.3}
            | VIDATA
            | {"Vao": Vao, "Vbo": Vbo, "Ebo": Ebo}
        )

        OneOfTheCube: P_Object = P_Object(**CData)
        self.Graphics.append(OneOfTheCube)

        index: int = len(self.Graphics)

        self.window_self.ObjectList.beginInsertRows(QModelIndex(), index, index)
        self.window_self.ObjectList.endInsertRows()

        self.update()
