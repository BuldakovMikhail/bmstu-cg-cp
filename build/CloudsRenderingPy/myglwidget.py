from clouds import Clouds

from PyQt5 import QtGui, QtWidgets, QtCore, QtOpenGL

import moderngl
import moderngl_window


class QModernGLWidget(QtOpenGL.QGLWidget):
    def __init__(self, *args):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        self.timer = QtCore.QElapsedTimer()
        super(QModernGLWidget, self).__init__(fmt, *args)

    def initializeGL(self):
        pass

    def paintGL(self):
        self.ctx = moderngl.create_context()
        moderngl_window.activate_context(ctx=self.ctx)
        self.screen = self.ctx.detect_framebuffer()
        self.init()
        self.render()
        self.paintGL = self.render

    def init(self):
        pass

    def render(self):
        pass


class MyWidget(QModernGLWidget):
    def __init__(self, *args):
        super(MyWidget, self).__init__(*args)
        self.scene = None
        self.t = 1

    def init(self):
        self.resize(*Clouds.window_size)
        self.ctx.viewport = (0, 0, *Clouds.window_size)
        self.scene = Clouds(self.ctx)

    def render(self):
        # print("render mywidget")
        self.screen.use()
        self.scene.clear()

        self.scene.render(self.t, 1)
        self.t += 0.5 / 10

    # def mousePressEvent(self, evt):
    #     pan_tool.start_drag(evt.x() / 512, evt.y() / 512)
    #     self.scene.pan(pan_tool.value)
    #     self.update()

    # def mouseMoveEvent(self, evt):
    #     pan_tool.dragging(evt.x() / 512, evt.y() / 512)
    #     self.scene.pan(pan_tool.value)
    #     self.update()

    # def mouseReleaseEvent(self, evt):
    #     pan_tool.stop_drag(evt.x() / 512, evt.y() / 512)
    #     self.scene.pan(pan_tool.value)
    #     self.update()
