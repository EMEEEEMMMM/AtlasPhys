from VisualPart import simulator
from OptionsUI import createUI
import multiprocessing as mp
from OpenGL.GLUT import *




def opengl_renderer(RenderQueue):
    renderer = simulator()
    
    def display():
        frame = renderer.render_frame()
        RenderQueue.put(frame)
        glutPostRedisplay()
    
    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutMainLoop()



def main():
    RenderQueue = mp.Queue(maxsize=1)
    RenderSizeQueue = mp.Queue(maxsize=2)
    KeysQueue = mp.Queue(maxsize=10)
    PyOpenGLProcess = mp.Process(
        target=opengl_renderer, args=(RenderQueue,)
    )
    UIProcess = mp.Process(target=createUI, args=(RenderQueue,))

    UIProcess.start()
    PyOpenGLProcess.start()
    UIProcess.join()
    PyOpenGLProcess.terminate()
    RenderQueue.close()
