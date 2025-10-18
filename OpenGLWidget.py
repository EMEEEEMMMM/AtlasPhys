from typing import Any, Union
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtCore import QPointF, Qt, QModelIndex
from OpenGL.GLU import gluLookAt
from OpenGL.GL import *
import numpy as np
import os
import sys
from numpy.typing import NDArray
from utils import Physics_utils

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
        self.EYE: NDArray[np.floating[Any]] = np.array([0.0, 0.0, 5.0])
        self.LOOK_AT: NDArray[np.floating[Any]] = np.array([0.0, 0.0, 0.0])
        self.EYE_UP: NDArray[np.floating[Any]] = np.array([0.0, 1.0, 0.0])

        self.SCALE_K: NDArray[np.floating[Any]] = np.array([1.0, 1.0, 1.0])
        self.IS_PERSPECTIVE: bool = True
        self.COORDINATE_AXIS: bool = False
        self.PLANE: bool = False
        self.VIEW: NDArray[np.floating[Any]] = np.array(
            [-0.8, 0.8, -0.8, 0.8, 1.0, 20.0]
        )

        self.RightDown: bool = False
        self.LastPos: QPointF = QPointF()
        self.Graphics: list = []
        self.Coordinate_Data = None
        self.Plane_Data = None

    def initializeGL(self):
        self.fmt = QSurfaceFormat()
        self.fmt.setVersion(3, 3)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(
            GL_DEPTH_TEST
        )  # Enable depth testing to achieve occlusion relationships
        self.shader_program = self.compile_shaders()
        glDepthFunc(
            GL_LEQUAL
        )  # Set the depth test function (GL_LEQUAL is just one of the options)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader_program)

    def paintGL(self):
        self.makeCurrent()
        w, h = self.width(), self.height()

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.IS_PERSPECTIVE:
            # Perspective Projection (glFrustum)
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
            glFrustum(left, right, bottom, top, self.VIEW[4], self.VIEW[5])
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
            glOrtho(left, right, bottom, top, self.VIEW[4], self.VIEW[5])

        Projection = glGetFloatv(GL_PROJECTION_MATRIX)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        gluLookAt(
            self.EYE[0],
            self.EYE[1],
            self.EYE[2],
            self.LOOK_AT[0],
            self.LOOK_AT[1],
            self.LOOK_AT[2],
            self.EYE_UP[0],
            self.EYE_UP[1],
            self.EYE_UP[2],
        )

        glScalef(self.SCALE_K[0], self.SCALE_K[1], self.SCALE_K[2])

        ModelView = glGetFloatv(GL_MODELVIEW_MATRIX)

        mvp = np.dot(Projection, ModelView)
        mvp_loc = glGetUniformLocation(self.shader_program, "mvp")
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp)

        for vao, _, _, index_len, object_type in self.Graphics:
            glBindVertexArray(vao)
            glDrawElements(object_type, index_len, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            glBindVertexArray(0)

        # Clear the buffer and send the instructions to the hardware for immediate execution
        glFlush()

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.RightButton:
            self.RightDown = True
            self.LastPos = event.position()

    def mouseMoveEvent(self, event):
        if not self.RightDown:
            return

        # Get the position of the mouse and its displacement
        Current_Position = event.position()
        dx = self.LastPos.x() - Current_Position.x()
        dy = Current_Position.y() - self.LastPos.y()
        self.LastPos = Current_Position

        # Calculate the vector from the current camera to the target point
        DirectionOfEye: NDArray[np.floating[Any]] = self.LOOK_AT - self.EYE
        Distance: np.floating[Any] = np.linalg.norm(DirectionOfEye)
        if Distance < 1e-6:
            return

        Theta: float = -dx * 0.002
        Phi: float = -dy * 0.002

        Phi = np.clip(Phi, -np.pi / 2.0 + 0.01, np.pi / 2.0 - 0.01)

        # Calculate the position of the camera rotated
        # Unit Vector
        UnitVector: NDArray[np.floating[Any]] = DirectionOfEye / Distance
        UP_Vector: NDArray[np.floating[Any]] = self.EYE_UP

        # Right
        Right: NDArray[np.floating[Any]] = np.cross(UnitVector, UP_Vector)
        Right_norm: np.floating = np.linalg.norm(Right)
        if Right_norm < 1e-6:
            Right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            Right = Right / Right_norm

        # Apply Vertical rotation
        RotatedVerticalVector: NDArray[np.floating[Any]] = Physics_utils.rotate_vector(
            UnitVector, UP_Vector, Theta
        )

        Right_2nd = np.cross(RotatedVerticalVector, UP_Vector)
        Right_norm_2nd = np.linalg.norm(Right_2nd)
        if Right_norm_2nd < 1e-6:
            Right_2nd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            Right_2nd = Right_2nd / Right_norm_2nd

        # Apply horizontal rotaion (The final vector)
        FinalVector: NDArray[np.floating[Any]] = Physics_utils.rotate_vector(
            RotatedVerticalVector, Right_2nd, Phi
        )
        FinalUP: NDArray[np.floating[Any]] = Physics_utils.rotate_vector(
            UP_Vector, Right_2nd, Phi
        )

        # Update self.EYE, self.EYE_UP
        self.EYE = self.LOOK_AT - FinalVector * Distance
        self.EYE_UP = FinalUP / np.linalg.norm(FinalUP)

        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.RightDown = False

    def keyPressEvent(self, event):
        """
        Handle keyboard key events and achieve camera position movement

        Args:
            event (QKeyEvent)
        """

        Step: float = 0.1

        DirectionForward: NDArray[np.floating[Any]] = self.LOOK_AT - self.EYE
        Norm_DirectionForward: np.floating = np.linalg.norm(DirectionForward)
        if Norm_DirectionForward < 1e-6:
            return
        DirectionForward = DirectionForward / Norm_DirectionForward

        DirectionRight: NDArray[np.floating[Any]] = np.cross(
            DirectionForward, self.EYE_UP
        )
        Norm_DirectionRight: np.floating = np.linalg.norm(DirectionRight)
        if Norm_DirectionRight < 1e-6:
            DirectionRight = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            DirectionRight = DirectionRight / Norm_DirectionRight

        match event.key():

            case Qt.Key.Key_W:
                self.EYE += DirectionForward * Step
                self.LOOK_AT += DirectionForward * Step

            case Qt.Key.Key_S:
                self.EYE -= DirectionForward * Step
                self.LOOK_AT -= DirectionForward * Step

            case Qt.Key.Key_A:
                self.EYE -= DirectionRight * Step
                self.LOOK_AT -= DirectionRight * Step

            case Qt.Key.Key_D:
                self.EYE += DirectionRight * Step
                self.LOOK_AT += DirectionRight * Step

            case Qt.Key.Key_Q:
                self.EYE += self.EYE_UP * Step
                self.LOOK_AT += self.EYE_UP * Step

            case Qt.Key.Key_E:
                self.EYE -= self.EYE_UP * Step
                self.LOOK_AT -= self.EYE_UP * Step

        self.update()

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
            vertex_shader, self.load_shader("ShaderProgram/vertex_shader.vert")
        )
        glCompileShader(vertex_shader)

        # Vertex_shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(
            fragment_shader, self.load_shader("ShaderProgram/fragment_shader.frag")
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
            -500, 10, -500, 2/3, 1/3, 0, 1.0,
            500, 10, -500, 2/3, 1/3, 0, 1.0,
            500, -10, -500, 2/3, 1/3, 0, 1.0,
            -500, -10, -500, 2/3, 1/3, 0, 1.0,
            -500, 10, 500, 2/3, 1/3, 0, 1.0,
            500, 10, 500, 2/3, 1/3, 0, 1.0,
            500, -10, 500, 2/3, 1/3, 0, 1.0,
            -500, -10, 500, 2/3, 1/3, 0, 1.0,
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

    def load_shader(self, file_path: str):
        """_summary_
        Load shader from file_path

        Args:
            file_path (str): the file path

        Returns:
            the shader
        """
        if getattr(sys, "frozen", False):
            BasePth = sys._MEIPASS  # type: ignore[attr-defined]
        else:
            BasePth = os.path.abspath(".")
        dir = os.path.join(BasePth, file_path)
        with open(dir, "r") as f:
            return f.read()

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
