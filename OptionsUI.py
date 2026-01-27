from PyQt6.QtWidgets import *  # type: ignore
from PyQt6.QtCore import Qt, QModelIndex, QAbstractListModel, QObject
from OpenGLWidget import Simulator
from Handler import PushButtonEvent
from typing import Any, Optional
from utils import G_Object


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.init_ui()
        self.init_data()

    def init_ui(self) -> None:
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
        AddOrDeleteCoordinateAxis = QPushButton("Add/Delete Coordinate Axis")
        self.AddOrDeletePlane = QPushButton("Add/Delete the plane")
        StartOrStop = QPushButton("Start/Stop the simulator")
        LoadDemo = QPushButton("Load/Unload the demo")
        self.AddCube = QPushButton("Shortcut to add a cube")
        self.AddSphere = QPushButton("Shortcut to add a sphere")

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
        LeftSidebarLayout.addWidget(AddOrDeleteCoordinateAxis)
        LeftSidebarLayout.addWidget(self.AddOrDeletePlane)
        LeftSidebarLayout.addWidget(StartOrStop)
        LeftSidebarLayout.addWidget(LoadDemo)
        LeftSidebarLayout.addWidget(self.AddCube)
        LeftSidebarLayout.addWidget(self.AddSphere)

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

        SwitchP_Mode.clicked.connect(
            lambda: self.PushButtonEvent.switch_projection_mode()
        )
        AddObjects.clicked.connect(lambda: self.PushButtonEvent.add_object())
        AddOrDeleteCoordinateAxis.clicked.connect(
            lambda: self.PushButtonEvent.add_or_delelte_coordinate_axis()
        )
        self.AddOrDeletePlane.clicked.connect(
            lambda: self.PushButtonEvent.add_or_delelte_plane()
        )
        StartOrStop.clicked.connect(lambda: self.PushButtonEvent.start_or_stop())
        LoadDemo.clicked.connect(lambda: self.PushButtonEvent.load_or_reload_demo())
        DeleteObjectBtn.clicked.connect(lambda: self.PushButtonEvent.delete_object())

        self.AddCube.clicked.connect(lambda: self.OpenGLWindow.add_demo_cube())
        self.AddSphere.clicked.connect(lambda: self.OpenGLWindow.add_demo_sphere())

        self.ObjectListView.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.ObjectListView.setEditTriggers(QListView.EditTrigger.NoEditTriggers)

    def init_data(self) -> None:
        self.ObjectList = ListObjectModel(self.OpenGLWindow)
        self.ObjectListView.setModel(self.ObjectList)


class ListObjectModel(QAbstractListModel):
    OpenGLWindow: Simulator
    ObjectList: list[Any]

    def __init__(
        self, OpenGLWindow: Simulator, parent: Optional[QObject] = None
    ) -> None:
        super().__init__(parent)
        self.OpenGLWindow = OpenGLWindow
        self.ObjectList = self.OpenGLWindow.Graphics

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.ObjectList) if not parent.isValid() else 0

    def data(
        self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Optional[str]:
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            Shape: str = self.ObjectList[index.row()].Shape
            return f"物体：{Shape}"
        return None

    def delete_selected(self, index: QModelIndex) -> None:
        if index.isValid():
            self.beginRemoveRows(QModelIndex(), index.row(), index.row())
            obj: G_Object.P_Object = self.ObjectList[index.row()]

            match obj.Shape:

                case "CoordinateAxis":
                    self.OpenGLWindow.COORDINATE_AXIS = False
                    self.OpenGLWindow.delete_single_object(obj.VAO, obj.VBO, obj.EBO)
                    del self.ObjectList[index.row()]
                    self.endRemoveRows()

                case "Plane":
                    self.OpenGLWindow.PLANE = False
                    self.OpenGLWindow.delete_single_object(obj.VAO, obj.VBO, obj.EBO)
                    del self.ObjectList[index.row()]
                    self.endRemoveRows()

                case _:
                    self.OpenGLWindow.delete_single_object(obj.VAO, obj.VBO, obj.EBO)
                    del self.ObjectList[index.row()]
                    self.endRemoveRows()

    def clear_all(self) -> None:
        self.beginRemoveRows(QModelIndex(), 0, len(self.ObjectList) - 1)
        self.ObjectList.clear()
        self.endRemoveRows()
