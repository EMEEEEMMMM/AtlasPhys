from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *

class Simulator(QOpenGLWidget):
    def __init__(self, parent=None):
        super(Simulator, self).__init__(parent)
        self.setMinimumSize(800, 600)

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBegin(GL_TRIANGLES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-0.5, -0.5, 0.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.5, -0.5, 0.0)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.5, 0.0)
        glEnd()