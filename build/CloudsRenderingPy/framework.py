from PyQt5 import QtWidgets
from PyQt5 import QtTest
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from gui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Initial settings
        self.setupUi(self)
        # super(MainWindow, self).showMaximized()
        timer = QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.openGLWidget.updateGL)
        timer.start()

        self.pushButton.clicked.connect(self.update_scene)

    def update_scene(self):
        sun_pos = self.x_sun.value(), self.y_sun.value(), self.z_sun.value()
        look = (
            self.x_look.value(),
            self.y_look.value(),
            self.z_look.value(),
        )

        self.openGLWidget.scene.safe_uniform("u_sun_pos", sun_pos)
        self.openGLWidget.scene.safe_uniform("u_look_at", look)
