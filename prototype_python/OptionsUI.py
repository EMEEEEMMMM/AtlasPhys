from PyQt6.QtWidgets import *  # type: ignore
from PyQt6.QtCore import Qt, QModelIndex, QAbstractListModel, QObject, pyqtSignal
from OpenGLWidget import Simulator
from Handler import Events
from typing import Any, Optional

from utils.G_Object import P_Object


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
        self.Gravity_Label = QLabel("Gravity:")
        Gravity = FloatSlider(parent=LeftSidebar)
        Gravity.setRange(-100 * Gravity.multiplier(), 0)
        Gravity.setFloatValue(-9.8)
        Gravity.setTickPosition(QSlider.TickPosition.TicksAbove)

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

        self.MA_Arrow = QCheckBox("MA Arrow")
        self.Impulse_Arrow = QCheckBox("Impulse Arrow")

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
        LeftSidebarLayout.addWidget(self.Gravity_Label)
        LeftSidebarLayout.addWidget(Gravity)
        # LeftSidebarLayout.addWidget(Slider2_Label)
        # LeftSidebarLayout.addWidget(Slider2)
        # LeftSidebarLayout.addWidget(Slider3_Label)
        # LeftSidebarLayout.addWidget(Slider3)
        # LeftSidebarLayout.addWidget(Slider4_Label)
        # LeftSidebarLayout.addWidget(Slider4)
        # LeftSidebarLayout.addWidget(Slider5_Label)
        # LeftSidebarLayout.addWidget(Slider5)

        LeftSidebarLayout.addWidget(self.MA_Arrow)
        LeftSidebarLayout.addWidget(self.Impulse_Arrow)

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

        self.Events = Events(self)

        SwitchP_Mode.clicked.connect(lambda: self.Events.switch_projection_mode())
        AddObjects.clicked.connect(lambda: self.Events.add_object())
        AddOrDeleteCoordinateAxis.clicked.connect(
            lambda: self.Events.add_or_delelte_coordinate_axis()
        )
        self.AddOrDeletePlane.clicked.connect(
            lambda: self.Events.add_or_delelte_plane()
        )
        StartOrStop.clicked.connect(lambda: self.Events.start_or_stop())
        LoadDemo.clicked.connect(lambda: self.Events.load_or_reload_demo())
        DeleteObjectBtn.clicked.connect(lambda: self.Events.delete_object())

        self.AddCube.clicked.connect(lambda: self.OpenGLWindow.add_demo_cube())
        self.AddSphere.clicked.connect(lambda: self.OpenGLWindow.add_demo_sphere())

        Gravity.valueChangedFloat.connect(
            lambda gravity_value: self.Events.set_gravity(gravity_value)  # type: ignore
        )


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
        self.DynamicObjects: list[P_Object] = self.OpenGLWindow.DynamicObjects

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
            obj: P_Object = self.ObjectList[index.row()]

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
                    self.OpenGLWindow.delete_arrows(obj)
                    self.DynamicObjects.remove(obj)
                    del self.ObjectList[index.row()]
                    self.endRemoveRows()

    def clear_all(self) -> None:
        self.beginRemoveRows(QModelIndex(), 0, len(self.ObjectList) - 1)
        self.DynamicObjects.clear()
        self.ObjectList.clear()
        self.endRemoveRows()


class FloatSlider(QSlider):
    valueChangedFloat = pyqtSignal(float)

    def __init__(self, parent=None) -> None:  # type: ignore
        super().__init__(Qt.Orientation.Horizontal, parent)  # type: ignore
        self._multiplier: int = 10

        self.valueChanged.connect(self.emitFloatValueChanged)

    def setFloatValue(self, floatValue: float) -> None:
        intValue = int(floatValue * self._multiplier)
        self.setValue(intValue)

    def floatValue(self) -> float:
        return self.value() / self._multiplier

    def emitFloatValueChanged(self) -> None:
        self.valueChangedFloat.emit(self.floatValue())

    def setMultiplier(self, multiplier: int) -> None:
        self._multiplier = multiplier

    def multiplier(self) -> int:
        return self._multiplier
