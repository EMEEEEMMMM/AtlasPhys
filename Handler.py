from typing import TypedDict, Any
from PyQt6.QtWidgets import *  # type: ignore
from PyQt6.QtCore import QModelIndex
import numpy as np
from numpy.typing import NDArray
from OpenGL.GL import *  # type: ignore
from utils import Generate_Objects, Opengl_utils
from utils.G_Object import P_Object
from utils.Decorator import time_counter


class ObjectDataType(TypedDict):
    Shape: str
    Side_Length: float
    X_Coordinate: float
    Y_Coordinate: float
    Z_Coordinate: float
    R_v: float
    G_v: float
    B_v: float
    A_v: float
    Mass: float
    Restitution: float


class PushButtonEvent:
    window: Any

    def __init__(self, window: Any) -> None:
        self.window = window

    @time_counter
    def switch_projection_mode(self) -> None:
        CurrentMode: bool = self.window.OpenGLWindow.IS_PERSPECTIVE
        if CurrentMode:
            self.window.OpenGLWindow.IS_PERSPECTIVE = False
            print(f"IS_PERSPECTIVE={self.window.OpenGLWindow.IS_PERSPECTIVE}")
        else:
            self.window.OpenGLWindow.IS_PERSPECTIVE = True
            print(f"IS_PERSPECTIVE={self.window.OpenGLWindow.IS_PERSPECTIVE}")
        self.window.OpenGLWindow.repaint()

    @time_counter
    def add_or_delelte_coordinate_axis(self) -> None:
        Current: bool = self.window.OpenGLWindow.COORDINATE_AXIS
        ObjectList: list[Any] = self.window.OpenGLWindow.Graphics
        if Current:
            Axis_Index: int = self.window.OpenGLWindow.Graphics.index(
                self.window.OpenGLWindow.CoordinateObj
            )
            obj = ObjectList[Axis_Index]
            self.window.OpenGLWindow.delete_single_object(obj.VAO, obj.VBO, obj.EBO)
            del ObjectList[Axis_Index]
            self.window.OpenGLWindow.COORDINATE_AXIS = False
        else:
            self.window.OpenGLWindow.draw_coordinates()
            self.window.OpenGLWindow.COORDINATE_AXIS = True

        self.window.OpenGLWindow.update()

    @time_counter
    def add_or_delelte_plane(self) -> None:
        Current: bool = self.window.OpenGLWindow.PLANE
        ObjectList: list[Any] = self.window.OpenGLWindow.Graphics
        if Current:
            Plane_Index: int = self.window.OpenGLWindow.Graphics.index(
                self.window.OpenGLWindow.PlaneObj
            )
            obj = ObjectList[Plane_Index]
            self.window.OpenGLWindow.delete_single_object(obj.VAO, obj.VBO, obj.EBO)
            del ObjectList[Plane_Index]
            self.window.OpenGLWindow.PLANE = False
        else:
            self.window.OpenGLWindow.draw_plane()
            self.window.OpenGLWindow.PLANE = True

        self.window.OpenGLWindow.update()

    @time_counter
    def add_object(self) -> None:
        dialog: AddObjectDialog = AddObjectDialog(self.window)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            data: ObjectDataType = dialog.get_data()
            print(data)

            Shape: str = data["Shape"]
            IData = {
                key: val
                for key, val in data.items()
                if key != "Shape" and key != "Mass" and key != "Restitution"
            }
            match Shape:  # type: ignore

                # case "Equilateral triangle":
                #     ObjectData: dict[
                #         str, int | NDArray[np.float32] | NDArray[np.uint32]
                #     ] = Generate_Objects.add_triangle(**IData)
                #     vao, vbo, ebo = Opengl_utils.analysis_data(
                #         self.window.OpenGLWindow, ObjectData["Vertices"], ObjectData["Indices"]  # type: ignore
                #     )
                #     CData: dict[str, object] = (
                #         data | ObjectData | {"Vao": vao, "Vbo": vbo, "Ebo": ebo}
                #     )
                #     Sustance: P_Object = P_Object(**CData)  # type: ignore
                #     self.window.OpenGLWindow.Graphics.append(Sustance)

                #     index: int = len(self.window.OpenGLWindow.Graphics)

                #     self.window.OpenGLWindow.window_self.ObjectList.beginInsertRows(
                #         QModelIndex(), index, index
                #     )
                #     self.window.OpenGLWindow.window_self.ObjectList.endInsertRows()

                # case "Square":
                #     ObjectData: dict[
                #         str, int | NDArray[np.float32] | NDArray[np.uint32]
                #     ] = Generate_Objects.add_square(**IData)
                #     vao, vbo, ebo = Opengl_utils.analysis_data(
                #         self.window.OpenGLWindow, ObjectData["Vertices"], ObjectData["Indices"]  # type: ignore
                #     )
                #     CData: dict[str, object] = (
                #         data | ObjectData | {"Vao": vao, "Vbo": vbo, "Ebo": ebo}
                #     )
                #     Sustance: P_Object = P_Object(**CData)  # type: ignore
                #     self.window.OpenGLWindow.Graphics.append(Sustance)

                #     index: int = len(self.window.OpenGLWindow.Graphics)

                #     self.window.OpenGLWindow.window_self.ObjectList.beginInsertRows(
                #         QModelIndex(), index, index
                #     )
                #     self.window.OpenGLWindow.window_self.ObjectList.endInsertRows()

                case "Cube":
                    ObjectData: dict[
                        str, int | NDArray[np.float32] | NDArray[np.uint32]
                    ] = Generate_Objects.add_cube(**IData)
                    vao, vbo, ebo = Opengl_utils.analysis_data(
                        self.window.OpenGLWindow, ObjectData["Vertices"], ObjectData["Indices"]  # type: ignore
                    )
                    CData: dict[str, object] = (
                        data | ObjectData | {"Vao": vao, "Vbo": vbo, "Ebo": ebo}
                    )
                    print(CData)
                    Sustance: P_Object = P_Object(**CData)  # type: ignore
                    self.window.OpenGLWindow.Graphics.append(Sustance)
                    self.window.OpenGLWindow.DynamicObjects.append(Sustance)

                    index: int = len(self.window.OpenGLWindow.Graphics)

                    self.window.OpenGLWindow.window_self.ObjectList.beginInsertRows(
                        QModelIndex(), index, index
                    )
                    self.window.OpenGLWindow.window_self.ObjectList.endInsertRows()

                case "Sphere":
                    ObjectData: dict[
                        str, int | NDArray[np.float32] | NDArray[np.uint32]
                    ] = Generate_Objects.add_sphere(**IData)
                    vao, vbo, ebo = Opengl_utils.analysis_data(
                        self.window.OpenGLWindow, ObjectData["Vertices"], ObjectData["Indices"]  # type: ignore
                    )
                    CData: dict[str, object] = (
                        data | ObjectData | {"Vao": vao, "Vbo": vbo, "Ebo": ebo}
                    )
                    Sustance: P_Object = P_Object(**CData)  # type: ignore
                    self.window.OpenGLWindow.Graphics.append(Sustance)
                    self.window.OpenGLWindow.DynamicObjects.append(Sustance)

                    index: int = len(self.window.OpenGLWindow.Graphics)

                    self.window.OpenGLWindow.window_self.ObjectList.beginInsertRows(
                        QModelIndex(), index, index
                    )
                    self.window.OpenGLWindow.window_self.ObjectList.endInsertRows()

    def start_or_stop(self) -> None:
        if self.window.OpenGLWindow.START_OR_STOP:
            self.window.OpenGLWindow.START_OR_STOP = False

        else:
            self.window.OpenGLWindow.start_render()
            self.window.OpenGLWindow.START_OR_STOP = True

    def load_or_reload_demo(self) -> None:
        if self.window.OpenGLWindow.DemoLoaded:
            self.window.OpenGLWindow.unload_demo()

        else:
            self.window.OpenGLWindow.load_demo()

    def delete_object(self) -> None:
        SelectedIndex: Any = self.window.ObjectListView.currentIndex()
        if SelectedIndex.isValid():
            self.window.ObjectList.delete_selected(SelectedIndex)
        self.window.update()


class AddObjectDialog(QDialog):
    r: float
    g: float
    b: float
    a: float
    SelectType: QComboBox
    Side_Length: QDoubleSpinBox
    X_Coordinate: QDoubleSpinBox
    Y_Coordinate: QDoubleSpinBox
    Z_Coordinate: QDoubleSpinBox
    ColorSelector: QPushButton
    ColorResult: QLabel
    Accept_Reject: QDialogButtonBox
    CurrentColor: tuple[float, float, float, float]

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Object")
        self.resize(600, 500)
        self.r = 0.0
        self.g = 0.0
        self.b = 0.0
        self.a = 0.0
        self.CurrentColor = (0.0, 0.0, 0.0, 1.0)
        self.init_ui()

    def init_ui(self) -> None:
        Mainwidget: QWidget = QWidget()
        MainLayout: QHBoxLayout = QHBoxLayout(Mainwidget)

        self.SelectType = QComboBox()
        self.SelectType.addItems(["Cube", "Sphere"])

        self.Side_Length = QDoubleSpinBox()
        self.Side_Length.setRange(0, 10)
        self.Side_Length.setValue(2)
        self.Side_Length.setDecimals(2)

        self.X_Coordinate = QDoubleSpinBox()
        self.X_Coordinate.setRange(-100, 100)
        self.X_Coordinate.setValue(0)
        self.X_Coordinate.setDecimals(2)

        self.Y_Coordinate = QDoubleSpinBox()
        self.Y_Coordinate.setRange(-100, 100)
        self.Y_Coordinate.setValue(10)
        self.Y_Coordinate.setDecimals(2)

        self.Z_Coordinate = QDoubleSpinBox()
        self.Z_Coordinate.setRange(-100, 100)
        self.Z_Coordinate.setValue(0)
        self.Z_Coordinate.setDecimals(2)

        self.ColorSelector = QPushButton("Select Color")
        self.ColorResult = QLabel(
            f"RGBA: r={self.r}, g={self.g}, b={self.b}, a={self.a}"
        )

        self.Object_Mass = QDoubleSpinBox()
        self.Object_Mass.setRange(0, 100)
        self.Object_Mass.setValue(1)
        self.Object_Mass.setDecimals(2)

        self.Object_Restitution = QDoubleSpinBox()
        self.Object_Restitution.setRange(0, 1)
        self.Object_Restitution.setValue(0.3)
        self.Object_Restitution.setDecimals(2)

        self.Accept_Reject = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )

        MainLayout.addWidget(self.SelectType)
        MainLayout.addWidget(self.Side_Length)
        MainLayout.addWidget(self.X_Coordinate)
        MainLayout.addWidget(self.Y_Coordinate)
        MainLayout.addWidget(self.Z_Coordinate)
        MainLayout.addWidget(self.ColorSelector)
        MainLayout.addWidget(self.ColorResult)
        MainLayout.addWidget(self.Object_Mass)
        MainLayout.addWidget(self.Object_Restitution)
        MainLayout.addWidget(self.Accept_Reject)

        self.setLayout(MainLayout)

        self.ColorSelector.clicked.connect(self.show_color_dialog)
        self.Accept_Reject.accepted.connect(self.accept)
        self.Accept_Reject.rejected.connect(self.reject)

    def get_data(self) -> ObjectDataType:
        return {
            "Shape": self.SelectType.currentText(),
            "Side_Length": self.Side_Length.value(),
            "X_Coordinate": self.X_Coordinate.value(),
            "Y_Coordinate": self.Y_Coordinate.value(),
            "Z_Coordinate": self.Z_Coordinate.value(),
            "R_v": self.r,
            "G_v": self.g,
            "B_v": self.b,
            "A_v": self.a,
            "Mass": self.Object_Mass.value(),
            "Restitution": self.Object_Restitution.value(),
        }

    def show_color_dialog(self) -> None:
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b, a = color.getRgb()
            self.r, self.g, self.b, self.a = r / 255.0, g / 255.0, b / 255.0, a / 255.0
            self.ColorResult.setText(
                f"RGBA: r={self.r}, g={self.g}, b={self.b}, a={self.a}"
            )

            self.CurrentColor = (self.r, self.g, self.b, self.a)
        else:
            self.CurrentColor = (0.0, 0.0, 0.0, 1.0)
