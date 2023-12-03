from PyQt5 import QtWidgets
from PyQt5 import QtTest
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from gui import Ui_MainWindow
from PyQt5.QtWidgets import QMessageBox


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

        if sun_pos[0] == 0 and sun_pos[1] == 0 and sun_pos[2] == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка")
            msg.setInformativeText(
                "Направление вектора падения света задано неправильно, установлено значение по умолчанию"
            )
            msg.setWindowTitle("Ошибка")
            msg.exec_()

            sun_pos = (0, 0.5, 1)

        look = (
            self.x_look.value(),
            self.y_look.value(),
            self.z_look.value() + 0.0001,
        )

        density = self.densitySB.value()
        coverage = self.coverageSB.value()
        eccentrisy = self.eccentrisySB.value()

        self.openGLWidget.scene.safe_uniform("u_sun_pos", sun_pos)
        self.openGLWidget.scene.safe_uniform("u_look_at", look)

        self.openGLWidget.scene.safe_uniform("u_density", density)
        self.openGLWidget.scene.safe_uniform("u_coverage", coverage)
        self.openGLWidget.scene.safe_uniform("u_eccentrisy", eccentrisy)
