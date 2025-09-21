from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QOpenGLVersionProfile
from PyQt5.QtCore import QPoint, Qt
from OpenGL.GLU import gluLookAt
from OpenGL.GL import *
import numpy as np

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
    def __init__(self, parent=None):
        super().__init__(parent)
        # Camera
        self.EYE = np.array([0.0, 0.0, 5.0])
        self.LOOK_AT = np.array([0.0, 0.0, 0.0])
        self.EYE_UP = np.array([0.0, 1.0, 0.0])

        self.DIST, self.PHI, self.THETA = self.get_posture()

        self.SCALE_K = np.array([1.0, 1.0, 1.0])
        self.IS_PERSPECTIVE = True
        self.VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])

        self.LeftDown = False
        self.LastPos = QPoint()

    def initializeGL(self):
        version_profile = QOpenGLVersionProfile()
        version_profile.setVersion(2, 0)
        self.gl = self.context().versionFunctions(version_profile)
        self.gl.initializeOpenGLFunctions()

        self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.gl.glEnable(
            self.gl.GL_DEPTH_TEST
        )  # Enable depth testing to achieve occlusion relationships
        self.gl.glDepthFunc(
            self.gl.GL_LEQUAL
        )  # Set the depth test function (GL_LEQUAL is just one of the options)
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        self.gl.glMatrixMode(self.gl.GL_PROJECTION)
        self.gl.glLoadIdentity()

    def paintGL(self):
        self.makeCurrent()
        w, h = self.width(), self.height()
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
            self.gl.glFrustum(left, right, bottom, top, self.VIEW[4], self.VIEW[5])
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
            self.gl.glOrtho(left, right, bottom, top, self.VIEW[4], self.VIEW[5])

        self.gl.glMatrixMode(self.gl.GL_MODELVIEW)
        self.gl.glLoadIdentity()

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

        self.gl.glScalef(self.SCALE_K[0], self.SCALE_K[1], self.SCALE_K[2])

        self.draw_triangle()
        # Clear the buffer and send the instructions to the hardware for immediate execution
        self.gl.glFlush()

    def get_posture(self):
        delta = self.EYE - self.LOOK_AT
        DIST = np.linalg.norm(delta)
        if DIST < 0.001:
            return 0.0, 0.0, 0.0

        PHI = np.arcsin(delta[1] / DIST)
        r = np.linalg.norm(delta[[0, 2]])
        THETA = np.arcsin(delta[0] / r) if r > 0.001 else 0.0

        return DIST, PHI, THETA

    def draw_triangle(self):
        # Start drawing world coordinates
        self.gl.glBegin(self.gl.GL_LINES)

        # Draw X-axis with color red
        self.gl.glColor4f(1.0, 0.0, 0.0, 1.0)  # set color red and non-transparent
        self.gl.glVertex3f(-0.8, 0.0, 0.0)  # X-axis vertex (negative X-axis)
        self.gl.glVertex3f(0.8, 0.0, 0.0)  # X-axis vertex (positive X-axis)

        # Draw Y-axis with color green
        self.gl.glColor4f(0.0, 1.0, 0.0, 1.0)
        self.gl.glVertex3f(0.0, -0.8, 0.0)
        self.gl.glVertex3f(0.0, 0.8, 0.0)

        # Draw Z-axis with color blue
        self.gl.glColor4f(0.0, 0.0, 1.0, 1.0)
        self.gl.glVertex3f(0.0, 0.0, -0.8)
        self.gl.glVertex3f(0.0, 0.0, 0.8)

        self.gl.glEnd()  # End

        # Start draw triangle
        self.gl.glBegin(self.gl.GL_TRIANGLES)

        self.gl.glColor4f(1.0, 0.0, 0.0, 1.0)
        self.gl.glVertex3f(-0.5, -0.366, -0.5)
        self.gl.glColor4f(0.0, 1.0, 0.0, 1.0)
        self.gl.glVertex3f(0.5, -0.366, -0.5)
        self.gl.glColor4f(0.0, 0.0, 1.0, 1.0)
        self.gl.glVertex3f(0.0, 0.5, -0.5)

        self.gl.glEnd()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.LeftDown = True
            self.LastPos = event.pos()

    def mouseMoveEvent(self, event):
        if not self.LeftDown:
            return

        dx = self.LastPos.x() - event.x()
        dy = event.y() - self.LastPos.y()
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
        if event.button() == Qt.LeftButton:
            self.LeftDown = False

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.SCALE_K *= 1.05
        else:
            self.SCALE_K *= 0.95

        self.repaint()

    def resizeGL(self, w, h):
        self.gl.glViewport(0, 0, w, h)
        self.update()
