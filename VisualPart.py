import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class simulator:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.angle = 0.0
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(b"Simulator")
        self.init_gl()

    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width/self.height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glutHideWindow()
    
    def render_frame(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        
        glRotatef(self.angle, 1, 1, 1)
        self.angle += 0.5
        
        glBegin(GL_QUADS)
        glColor3f(1.0, 0.0, 0.0) 
        glVertex3f(-1, -1, 1)
        glVertex3f(1, -1, 1)
        glVertex3f(1, 1, 1)
        glVertex3f(-1, 1, 1)
        
        glColor3f(0.0, 1.0, 0.0)  
        glVertex3f(-1, -1, -1)
        glVertex3f(1, -1, -1)
        glVertex3f(1, 1, -1)
        glVertex3f(-1, 1, -1)
        

        glEnd()
        
        glutSwapBuffers()
        return self.capture_frame()
    
    def capture_frame(self):
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
        return np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
