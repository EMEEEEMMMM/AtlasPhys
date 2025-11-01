from typing import Any
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtCore import QPointF, Qt, QModelIndex, QElapsedTimer, QTimer
from OpenGL.GL import *
import numpy as np
import os
import sys
from numpy.typing import NDArray
from utils import Opengl_utils, Physics_utils

# IS_PERSPECTIVE = True                               # Perspective projection
# VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # Visual body left/right/top/bottom/near/far six surface
# SCALE_K = np.array([1.0, 1.0, 1.0])                 # Model scaling ratio
# EYE = np.array([0.0, 0.0, 2.0])                     # The position of the eyes (default positive Direction of the Z-axis)
# LOOK_AT = np.array([0.0, 0.0, 0.0])                 # Reference point for aiming Direction (default at the coordinate origin)
# EYE_UP = np.array([0.0, 1.0, 0.0])                  # Define the top (default positive Direction of the Y-axis) for the observer
# WIN_W, WIN_H = 640, 480                             # Variables for saving the width and height of the window
# LEFT_IS_DOWNED = False                              # The left mouse button was pressed
# MOUSE_X, MOUSE_Y = 0, 0                             # The starting position saved when examining the mouse displacement


class Simulator(QOpenGLWidget):
    def __init__(self, window_ui, parent=None) -> None:
        super().__init__(parent)

        self.window_self = window_ui

        # Camera
        self.CameraPos: NDArray[np.floating[Any]] = np.array(
            [0.0, 0.0, 3.0], dtype=np.float32
        )
        self.CameraFront: NDArray[np.floating[Any]] = np.array(
            [0.0, 0.0, -1.0], dtype=np.float32
        )
        self.CameraUP: NDArray[np.floating[Any]] = np.array(
            [0.0, 1.0, 0.0], dtype=np.float32
        )

        self.SCALE_K: NDArray[np.floating[Any]] = np.array([1.0, 1.0, 1.0])
        self.IS_PERSPECTIVE: bool = True
        self.COORDINATE_AXIS: bool = False
        self.PLANE: bool = False
        self.VIEW: NDArray[np.floating[Any]] = np.array(
            [-0.8, 0.8, -0.8, 0.8, 1.0, 20.0]
        )
        self.YAW: float = -90.0
        self.PITCH: float = 0.0

        self.RightDown: bool = False
        self.LastPos: QPointF = QPointF()
        self.Graphics: list = []
        self.Coordinate_Data = None
        self.Plane_Data = None

        self.TIMER = QElapsedTimer()
        self.TIMER_STARTED: bool = False
        self.KEYSPRESSED: list = []
        self.Animation_Timer = QTimer(self)
        self.Animation_Timer.timeout.connect(self.update)
        self.Animation_Timer.start(7)

    def initializeGL(self):
        self.fmt = QSurfaceFormat()
        self.fmt.setVersion(3, 3)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        # Enable depth testing to achieve occlusion relationships
        self.shader_program = self.compile_shaders()
        glDepthFunc(GL_LEQUAL)
        # Set the depth test function (GL_LEQUAL is just one of the options)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw_plane()
        # self.draw_sun()

    def paintGL(self):
        self.makeCurrent()
        w, h = self.width(), self.height()

        if self.IS_PERSPECTIVE:
            # Perspective Projection (glFrustum)
            if w > h:
                left: np.float64 = self.VIEW[0] * w / h
                right: np.float64 = self.VIEW[1] * w / h
                bottom: np.float64 = self.VIEW[2]
                top: np.float64 = self.VIEW[3]
            else:
                left: np.float64 = self.VIEW[0]
                right: np.float64 = self.VIEW[1]
                bottom: np.float64 = self.VIEW[2] * h / w
                top: np.float64 = self.VIEW[3] * h / w
            projection: NDArray[np.floating[Any]] = Opengl_utils.perspective_projection(
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

        ViewMatrix: NDArray[np.floating[Any]] = Opengl_utils.lookat(
            self.CameraPos, self.CameraPos + self.CameraFront, self.CameraUP
        )
        ModelMatrix: NDArray[np.floating[Any]] = Opengl_utils.scalef(self.SCALE_K)

        glUseProgram(self.shader_program)

        ProjLocation = glGetUniformLocation(self.shader_program, "projection")
        ViewLocation = glGetUniformLocation(self.shader_program, "view")
        ModelLocation = glGetUniformLocation(self.shader_program, "model")

        glUniformMatrix4fv(ProjLocation, 1, GL_FALSE, projection.T.flatten())
        glUniformMatrix4fv(ViewLocation, 1, GL_FALSE, ViewMatrix.T.flatten())
        glUniformMatrix4fv(ModelLocation, 1, GL_FALSE, ModelMatrix.T.flatten())

        LightColor = glGetUniformLocation(self.shader_program, "lightColor")
        glUniform4f(LightColor, 1.0, 1.0, 1.0, 1.0)

        for vao, _, _, index_len, object_type in self.Graphics:
            glBindVertexArray(vao)
            glDrawElements(object_type, index_len, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            glBindVertexArray(0)

        # Clear the buffer and send the instructions to the hardware for immediate execution
        glFlush()

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        if event.button() == Qt.MouseButton.RightButton:
            self.RightDown = True
            self.LastPos = event.position()

            front = self.CameraFront
            self.YAW = np.degrees(np.arctan2(front[2], front[0]))
            self.PITCH = np.degrees(np.arcsin(np.clip(front[1], -1.0, 1.0)))

    def mouseMoveEvent(self, event) -> None:
        if not self.RightDown:
            return

        Current_Position = event.position()
        DX = Current_Position.x() - self.LastPos.x()
        DY = self.LastPos.y() - Current_Position.y()

        Sensitivity: float = 0.05
        self.YAW += DX * Sensitivity
        self.PITCH += DY * Sensitivity

        self.PITCH = np.clip(self.PITCH, -89.0, 89.0)

        Front: NDArray[np.floating[Any]] = np.array(
            [
                np.cos(np.radians(self.YAW)) * np.cos(np.radians(self.PITCH)),
                np.sin(np.radians(self.PITCH)),
                np.sin(np.radians(self.YAW)) * np.cos(np.radians(self.PITCH)),
            ], dtype=np.float32
        )
        self.CameraFront = Physics_utils.normalize(Front)

        self.LastPos = Current_Position

        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.RightDown = False

    def keyPressEvent(self, event):
        CameraSpeed: float = 5.0 * 1 / 144

        match event.key():

            case Qt.Key.Key_W:
                self.CameraPos += CameraSpeed * self.CameraFront

            case Qt.Key.Key_S:
                self.CameraPos -= CameraSpeed * self.CameraFront

            case Qt.Key.Key_A:
                self.CameraPos -= (
                    Physics_utils.normalize(np.cross(self.CameraFront, self.CameraUP))
                    * CameraSpeed
                )

            case Qt.Key.Key_D:
                self.CameraPos += (
                    Physics_utils.normalize(np.cross(self.CameraFront, self.CameraUP))
                    * CameraSpeed
                )

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.SCALE_K *= 1.05
        else:
            self.SCALE_K *= 0.95

        self.update()

    def resizeGL(self, w, h):
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

        Indices: NDArray[np.float32] = np.array([
            0, 1,
            2, 3,
            4, 5
        ], dtype=np.uint32)
        # fmt: on

        # Vao
        self.AxisVao = glGenVertexArrays(1)
        glBindVertexArray(self.AxisVao)

        # Vbo
        self.AxisVbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.AxisVbo)
        glBufferData(GL_ARRAY_BUFFER, Vertices.nbytes, Vertices, GL_STATIC_DRAW)

        # Ebo
        self.AxisEbo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.AxisEbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, Indices.nbytes, Indices, GL_STATIC_DRAW)

        # Vertices
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        index = len(self.Graphics)
        self.Coordinate_Data = (
            self.AxisVao,
            self.AxisVbo,
            self.AxisEbo,
            len(Indices),
            GL_LINES,
        )
        self.Graphics.append(
            (self.AxisVao, self.AxisVbo, self.AxisEbo, len(Indices), GL_LINES)
        )

        self.window_self.ObjectList.beginInsertRows(QModelIndex(), index, index)
        self.window_self.ObjectList.endInsertRows()

        self.update()

        # glBindVertexArray(vao)
        # glDrawElements(GL_LINES, len(Indices), GL_UNSIGNED_INT, ctypes.c_void_p(0))
        # glBindVertexArray(0)
        # # Start drawing world coordinates
        # glBegin(GL_LINES)

        # # Draw X-axis with color red
        # glColor4f(1.0, 0.0, 0.0, 1.0)  # set color red and non-transparent
        # glVertex3f(-0.8, 0.0, 0.0)  # X-axis vertex (negative X-axis)
        # glVertex3f(0.8, 0.0, 0.0)  # X-axis vertex (positive X-axis)

        # # Draw Y-axis with color green
        # glColor4f(0.0, 1.0, 0.0, 1.0)
        # glVertex3f(0.0, -0.8, 0.0)
        # glVertex3f(0.0, 0.8, 0.0)

        # # Draw Z-axis with color blue
        # glColor4f(0.0, 0.0, 1.0, 1.0)
        # glVertex3f(0.0, 0.0, -0.8)
        # glVertex3f(0.0, 0.0, 0.8)

        # glEnd()  # End

    def draw_plane(self) -> None:
        """
        To generate a plane on the screen
        """
        self.makeCurrent()

        # fmt: off
        Vertices: NDArray[np.float32] = np.array([
            -50, -2.0, -50, 2/3, 1/3, 0, 1.0,
            50, -2.0, -50, 2/3, 1/3, 0, 1.0,
            50, -2.05, -50, 2/3, 1/3, 0, 1.0,
            -50, -2.05, -50, 2/3, 1/3, 0, 1.0,
            -50, -2.0, 50, 2/3, 1/3, 0, 1.0,
            50, -2.0, 50, 2/3, 1/3, 0, 1.0,
            50, -2.05, 50, 2/3, 1/3, 0, 1.0,
            -50, -2.05, 50, 2/3, 1/3, 0, 1.0,
        ], dtype=np.float32)

        Indices: NDArray[np.uint32] = np.array([
            0, 1, 2, 3, # v0-v1-v2-v3
            4, 5, 1, 0, # v4-v5-v1-v0
            3, 2, 6, 7, # v3-v2-v6-v7
            5, 4, 7, 6, # v5-v4-v7-v6
            1, 5, 6, 2, # v1-v5-v6-v2
            4, 0, 3, 7  # v4-v0-v3-v7
        ], dtype=np.uint32)
        # fmt: on

        # Vao
        self.PlaneVao = glGenVertexArrays(1)
        glBindVertexArray(self.PlaneVao)

        # Vbo
        self.PlaneVbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.PlaneVbo)
        glBufferData(GL_ARRAY_BUFFER, Vertices.nbytes, Vertices, GL_STATIC_DRAW)

        # Ebo
        self.PlaneEbo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.PlaneEbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, Indices.nbytes, Indices, GL_STATIC_DRAW)

        # Vertices
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        index = len(self.Graphics)
        self.Plane_Data = (
            self.PlaneVao,
            self.PlaneVbo,
            self.PlaneEbo,
            len(Indices),
            GL_QUADS,
        )
        self.Graphics.append(
            (self.PlaneVao, self.PlaneVbo, self.PlaneEbo, len(Indices), GL_QUADS)
        )

        self.window_self.ObjectList.beginInsertRows(QModelIndex(), index, index)
        self.window_self.ObjectList.endInsertRows()

        self.update()

    def draw_sun(self) -> None:
        """
        Create a light for the entire environment


        TODO: A delete or add muti-light function
        """
        self.makeCurrent()
        Vertices, Indices = Opengl_utils.generate_sphere(
            5.0, 0.0, 10.0, 0.0, 10000, 10000
        )

        # Vao
        self.SunVao = glGenVertexArrays(1)
        glBindVertexArray(self.SunVao)

        # Vbo
        self.SunVbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.SunVbo)
        glBufferData(GL_ARRAY_BUFFER, Vertices.nbytes, Vertices, GL_STATIC_DRAW)

        # Ebo
        self.SunEbo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.SunEbo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, Indices.nbytes, Indices, GL_STATIC_DRAW)

        # Vertices
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        self.Graphics.append(
            (self.SunVao, self.SunVbo, self.SunEbo, len(Indices), GL_TRIANGLES)
        )

        self.update()

    def analysis_data(self, data: dict):
        """
        To analysis data were transfered from the Handler.py,
        bind the vao, vbo, ebo,
        and then send these info into self.Graphics to paint the object,

        Args:
            data (dict): the data of the object
        """
        self.makeCurrent()

        ObjectType: int = data["Type"]
        Vertices: NDArray[np.float32] = data["Vertices"]
        Indices: NDArray[np.uint64] = data["Indices"]

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, Vertices.nbytes, Vertices, GL_STATIC_DRAW)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, Indices.nbytes, Indices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        index = len(self.Graphics)

        self.Graphics.append((vao, vbo, ebo, len(Indices), ObjectType))

        self.window_self.ObjectList.beginInsertRows(QModelIndex(), index, index)
        self.window_self.ObjectList.endInsertRows()

        self.update()

    def delete_single_object(self, vao, vbo, ebo) -> None:
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
        if not self.TIMER_STARTED:
            self.TIMER.start()
            self.TIMER_STARTED = True

        return self.TIMER.elapsed() / 1000.0
