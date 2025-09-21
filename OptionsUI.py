from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from OpenGLWidget import Simulator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.InitUI()

    def InitUI(self):
        self.setWindowTitle("Main")
        self.setGeometry(100, 100, 800, 600)

        MainWidget = QWidget()
        MainLayout = QHBoxLayout(MainWidget)

        LeftSidebar = QWidget()
        LeftSidebarLayout = QVBoxLayout(LeftSidebar)
        LeftSidebarLayout.setAlignment(Qt.AlignTop)

        LeftSidebar_Label = QLabel("Options")
        btn1 = QPushButton("1")
        btn2 = QPushButton("2")
        btn3 = QPushButton("3")
        btn4 = QPushButton("4")

        LeftSidebarLayout.addWidget(LeftSidebar_Label)
        LeftSidebarLayout.addWidget(btn1)
        LeftSidebarLayout.addWidget(btn2)
        LeftSidebarLayout.addWidget(btn3)
        LeftSidebarLayout.addWidget(btn4)

        self.OpenGLWindow = Simulator()

        RightSideBar = QWidget()
        RightSideBarLayout = QVBoxLayout(RightSideBar)
        RightSideBarLayout.setAlignment(Qt.AlignTop)

        RightSideBar_Label = QLabel("Properties")
        Slider1_Label = QLabel("Slider 1")
        Slider1 = QSlider(Qt.Horizontal)
        Slider1.setMinimum(0)
        Slider1.setMaximum(100)
        Slider1.setValue(50)
        Slider2_Label = QLabel("Slider 2")
        Slider2 = QSlider(Qt.Horizontal)
        Slider2.setMinimum(0)
        Slider2.setMaximum(100)
        Slider2.setValue(50)
        Slider3_Label = QLabel("Slider 3")
        Slider3 = QSlider(Qt.Horizontal)
        Slider3.setMinimum(0)
        Slider3.setMaximum(100)
        Slider3.setValue(50)
        Slider4_Label = QLabel("Slider 4")
        Slider4 = QSlider(Qt.Horizontal)
        Slider4.setMinimum(0)
        Slider4.setMaximum(100)
        Slider4.setValue(50)
        Slider5_Label = QLabel("Slider 5")
        Slider5 = QSlider(Qt.Horizontal)
        Slider5.setMinimum(0)
        Slider5.setMaximum(100)
        Slider5.setValue(50)

        RightSideBarLayout.addWidget(RightSideBar_Label)
        RightSideBarLayout.addWidget(Slider1_Label)
        RightSideBarLayout.addWidget(Slider1)
        RightSideBarLayout.addWidget(Slider2_Label)
        RightSideBarLayout.addWidget(Slider2)
        RightSideBarLayout.addWidget(Slider3_Label)
        RightSideBarLayout.addWidget(Slider3)
        RightSideBarLayout.addWidget(Slider4_Label)
        RightSideBarLayout.addWidget(Slider4)
        RightSideBarLayout.addWidget(Slider5_Label)
        RightSideBarLayout.addWidget(Slider5)

        MainLayout.addWidget(LeftSidebar)
        MainLayout.addWidget(self.OpenGLWindow, 1)
        MainLayout.addWidget(RightSideBar)

        self.setCentralWidget(MainWidget)

