from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGL import QOpenGLVersionProfile, QOpenGLWindow
from PyQt6.QtCore import QPoint, Qt, QModelIndex
from OpenGL.GLU import gluLookAt
from OpenGL.GL import *
import numpy as np
import os
import sys
from numpy.typing import NDArray

# IS_PERSPECTIVE = True                               # Perspective projection
# VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # Visual body left/right/top/bottom/near/far six surface
# SCALE_K = np.array([1.0, 1.0, 1.0])                 # Model scaling ratio
# EYE = np.array([0.0, 0.0, 2.0])                     # The position of the eyes (default positive direction of the Z-axis)
# LOOK_AT = np.array([0.0, 0.0, 0.0])                 # Reference point for aiming direction (default at the coordinate origin)
# EYE_UP = np.array([0.0, 1.0, 0.0])                  # Define the top (default positive direction of the Y-axis) for the observer
# WIN_W, WIN_H = 640, 480                             # Variables for saving the width and height of the window
# LEFT_IS_DOWNED = False                              # The left mouse button was pressed
# MOUSE_X, MOUSE_Y = 0, 0                             # The starting position saved when examining the mouse displacement


class Simulator(QOpenGLWidget):
    def __init__(self, window_ui, parent=None):
        super().__init__(parent)

        self.window_self = window_ui
        # Camera
        self.EYE: NDArray = np.array([0.0, 0.0, 5.0])
        self.LOOK_AT: NDArray = np.array([0.0, 0.0, 0.0])
        self.EYE_UP: NDArray = np.array([0.0, 1.0, 0.0])

        self.DIST, self.PHI, self.THETA = self.get_posture()

        self.SCALE_K: NDArray = np.array([1.0, 1.0, 1.0])
        self.IS_PERSPECTIVE: bool = True
        self.VIEW: NDArray = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])

        self.LeftDown: bool = False
        self.LastPos = QPoint()
        self.Graphics: list = []

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
        # Set the projection matrix

    def paintGL(self):
        self.makeCurrent()
        w, h = self.width(), self.height()

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

        self.draw_coordinates()

        for vao, ebo, index_len, object_type in self.Graphics:
            glBindVertexArray(vao)
            glDrawElements(object_type, index_len, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            glBindVertexArray(0)

        # Clear the buffer and send the instructions to the hardware for immediate execution
        glFlush()

    def get_posture(self):
        delta: NDArray = self.EYE - self.LOOK_AT
        DIST = np.linalg.norm(delta)
        if DIST < 0.001:
            return 0.0, 0.0, 0.0

        PHI = np.arcsin(delta[1] / DIST)
        r = np.linalg.norm(delta[[0, 2]])
        THETA = np.arcsin(delta[0] / r) if r > 0.001 else 0.0

        return DIST, PHI, THETA

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.LeftDown = True
            self.LastPos = event.position()

    def mouseMoveEvent(self, event):
        if not self.LeftDown:
            return

        dx = self.LastPos.x() - event.position().x()
        dy = event.position().y() - self.LastPos.y()
        self.LastPos = event.pos()

        w, h = self.width(), self.height()
        self.PHI += 2 * np.pi * dy / h
        self.THETA += 2 * np.pi * dx / w

        self.PHI %= 2 * np.pi
        self.THETA %= 2 * np.pi

        r = self.DIST * np.cos(self.PHI)
        self.EYE[0] = self.LOOK_AT[0] + r * np.sin(self.THETA)
        self.EYE[1] = self.LOOK_AT[1] + self.DIST * np.sin(self.PHI)
        self.EYE[2] = self.LOOK_AT[2] + r * np.cos(self.THETA)

        if np.pi / 2 < self.PHI < 3 * np.pi / 2:
            self.EYE_UP[1] = -1.0
        else:
            self.EYE_UP[1] = 1.0

        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.LeftDown = False

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

    def draw_coordinates(self):
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
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # Vbo
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, Vertices.nbytes, Vertices, GL_STATIC_DRAW)

        # Ebo
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
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

        # self.Graphics.append((vao, ebo, len(indices), GL_LINES))
        glBindVertexArray(vao)
        glDrawElements(GL_LINES, len(Indices), GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)
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

    def analysis_data(self, data: dict):
        self.makeCurrent()

        ObjectType: int = data["Type"]
        Vertices: NDArray[np.float32] = data["Vertices"]
        Indices: NDArray[np.float32] = data["Indices"]

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

        self.Graphics.append((vao, ebo, len(Indices), ObjectType))

        self.window_self.ObjectList.beginInsertRows(QModelIndex(), index, index)
        self.window_self.ObjectList.endInsertRows()

        self.update()

    def load_shader(self, file_path: str):
        if getattr(sys, "frozen", False):
            BasePth = sys._MEIPASS
        else:
            BasePth = os.path.abspath(".")
        dir = os.path.join(BasePth, file_path)
        with open(dir, "r") as f:
            return f.read()
