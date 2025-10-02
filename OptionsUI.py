from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QModelIndex, QAbstractListModel
from OpenGLWidget import Simulator
from Handler import PushButtonEvent


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_data()

    def init_ui(self):
        self.setWindowTitle("Main")
        self.setGeometry(100, 100, 800, 600)

        MainWidget = QWidget()
        MainLayout = QHBoxLayout(MainWidget)

        LeftSidebar = QWidget()
        LeftSidebar.setFixedWidth(200)
        LeftSidebarLayout = QVBoxLayout(LeftSidebar)
        LeftSidebarLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        LeftSidebar_Label = QLabel("Options")
        SwitchP_Mode = QPushButton("Switch projection mode")
        AddObjects = QPushButton("Add Objects")
        btn3 = QPushButton("3")
        btn4 = QPushButton("4")

        Slider_Label = QLabel("Properties")
        Slider1_Label = QLabel("Slider 1")
        Slider1 = QSlider(Qt.Orientation.Horizontal)
        Slider1.setMinimum(0)
        Slider1.setMaximum(100)
        Slider1.setValue(50)
        Slider2_Label = QLabel("Slider 2")
        Slider2 = QSlider(Qt.Orientation.Horizontal)
        Slider2.setMinimum(0)
        Slider2.setMaximum(100)
        Slider2.setValue(50)
        Slider3_Label = QLabel("Slider 3")
        Slider3 = QSlider(Qt.Orientation.Horizontal)
        Slider3.setMinimum(0)
        Slider3.setMaximum(100)
        Slider3.setValue(50)
        Slider4_Label = QLabel("Slider 4")
        Slider4 = QSlider(Qt.Orientation.Horizontal)
        Slider4.setMinimum(0)
        Slider4.setMaximum(100)
        Slider4.setValue(50)
        Slider5_Label = QLabel("Slider 5")
        Slider5 = QSlider(Qt.Orientation.Horizontal)
        Slider5.setMinimum(0)
        Slider5.setMaximum(100)
        Slider5.setValue(50)

        LeftSidebarLayout.addWidget(LeftSidebar_Label)
        LeftSidebarLayout.addWidget(SwitchP_Mode)
        LeftSidebarLayout.addWidget(AddObjects)
        LeftSidebarLayout.addWidget(btn3)
        LeftSidebarLayout.addWidget(btn4)

        LeftSidebarLayout.addWidget(Slider_Label)
        LeftSidebarLayout.addWidget(Slider1_Label)
        LeftSidebarLayout.addWidget(Slider1)
        LeftSidebarLayout.addWidget(Slider2_Label)
        LeftSidebarLayout.addWidget(Slider2)
        LeftSidebarLayout.addWidget(Slider3_Label)
        LeftSidebarLayout.addWidget(Slider3)
        LeftSidebarLayout.addWidget(Slider4_Label)
        LeftSidebarLayout.addWidget(Slider4)
        LeftSidebarLayout.addWidget(Slider5_Label)
        LeftSidebarLayout.addWidget(Slider5)

        self.OpenGLWindow = Simulator(self)

        RightSidebar = QWidget()
        RightSidebar.setFixedWidth(200)
        RightSidebarLayout = QVBoxLayout(RightSidebar)
        RightSidebarLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.ObjectListView = QListView()
        DeleteObjectBtn = QPushButton("Delete")

        RightSidebarLayout.addWidget(self.ObjectListView)
        RightSidebarLayout.addWidget(DeleteObjectBtn)

        MainLayout.addWidget(LeftSidebar)
        MainLayout.addWidget(self.OpenGLWindow, 1)
        MainLayout.addWidget(RightSidebar)

        self.setCentralWidget(MainWidget)

        self.PushButtonEvent = PushButtonEvent(self)

        SwitchP_Mode.clicked.connect(self.PushButtonEvent.switch_projection_mode)
        AddObjects.clicked.connect(self.PushButtonEvent.add_object)

        DeleteObjectBtn.clicked.connect(self.PushButtonEvent.delete_object)
        self.ObjectListView.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.ObjectListView.setEditTriggers(QListView.EditTrigger.NoEditTriggers)

    def init_data(self):
        self.ObjectList: list = ListObjectModel(self.OpenGLWindow.Graphics)
        self.ObjectListView.setModel(self.ObjectList)


class ListObjectModel(QAbstractListModel):
    def __init__(self, Object_list: list, parent=None):
        super().__init__(parent)
        self.ObjectList = Object_list

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.ObjectList) if not parent.isValid() else 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            Type = self.ObjectList[index.row()][3]
            return f"物体：{Type}"

        return None

    def delete_selected(self, index: QModelIndex):
        if index.isValid():
            self.beginRemoveRows(QModelIndex(), index.row(), index.row())
            del self.ObjectList[index.row()]
            self.endRemoveRows()

            

