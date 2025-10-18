from typing import TypedDict
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QModelIndex
import numpy as np
from numpy.typing import NDArray
from typing import Union
from OpenGL.GL import *
from utils import Sphere_utils


class PushButtonEvent:
    def __init__(self, window) -> None:
        self.window = window

    def switch_projection_mode(self) -> None:
        CurrentMode = self.window.OpenGLWindow.IS_PERSPECTIVE
        if CurrentMode:
            self.window.OpenGLWindow.IS_PERSPECTIVE = False
            print(f"IS_PERSPECTIVE={self.window.OpenGLWindow.IS_PERSPECTIVE}")
        else:
            self.window.OpenGLWindow.IS_PERSPECTIVE = True
            print(f"IS_PERSPECTIVE={self.window.OpenGLWindow.IS_PERSPECTIVE}")
        self.window.OpenGLWindow.repaint()

    def add_or_delelte_coordinate_axis(self):
        Current: bool = self.window.OpenGLWindow.COORDINATE_AXIS
        ObjectList: list = self.window.OpenGLWindow.Graphics
        if Current:
            Axis_Index: int = self.window.OpenGLWindow.Graphics.index(
                (
                    self.window.OpenGLWindow.AxisVao,
                    self.window.OpenGLWindow.AxisVbo,
                    self.window.OpenGLWindow.AxisEbo,
                    6,
                    GL_LINES,
                )
            )
            (
                vao,
                vbo,
                ebo,
                _,
                _,
            ) = ObjectList[Axis_Index]
            self.window.OpenGLWindow.delete_single_object(vao, vbo, ebo)
            del ObjectList[Axis_Index]
            self.window.OpenGLWindow.COORDINATE_AXIS = False
        else:
            self.window.OpenGLWindow.draw_coordinates()
            self.window.OpenGLWindow.COORDINATE_AXIS = True

        self.window.OpenGLWindow.update()

    def add_or_delelte_plane(self):
        Current: bool = self.window.OpenGLWindow.PLANE
        ObjectList: list = self.window.OpenGLWindow.Graphics
        if Current:
            Plane_Index: int = self.window.OpenGLWindow.Graphics.index(
                (
                    self.window.OpenGLWindow.PlaneVao,
                    self.window.OpenGLWindow.PlaneVbo,
                    self.window.OpenGLWindow.PlaneEbo,
                    24,
                    GL_QUADS,
                )
            )
            (
                vao,
                vbo,
                ebo,
                _,
                _,
            ) = ObjectList[Plane_Index]
            self.window.OpenGLWindow.delete_single_object(vao, vbo, ebo)
            del ObjectList[Plane_Index]
            self.window.OpenGLWindow.PLANE = False
        else:
            self.window.OpenGLWindow.draw_plane()
            self.window.OpenGLWindow.PLANE = True

        self.window.OpenGLWindow.update()

    def add_object(self) -> None:
        dialog = AddObjectDialog(self.window)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            data: ObjectDataType = dialog.get_data()
            print(data)

        Type: str = data["Type"]
        Side_Length: float = data["Side_Length"]
        X_Coordinate: float = data["X_Coordinate"]
        Y_Coordinate: float = data["Y_Coordinate"]
        Z_Coordinate: float = data["Z_Coordinate"]
        R_v: float = data["R"]
        G_v: float = data["G"]
        B_v: float = data["B"]
        A_v: float = data["A"]

        match Type:
            case "Equilateral triangle":
                self.add_triangle(
                    Side_Length,
                    X_Coordinate,
                    Y_Coordinate,
                    Z_Coordinate,
                    R_v,
                    G_v,
                    B_v,
                    A_v,
                )

            case "Square":
                self.add_square(
                    Side_Length,
                    X_Coordinate,
                    Y_Coordinate,
                    Z_Coordinate,
                    R_v,
                    G_v,
                    B_v,
                    A_v,
                )

            case "Cube":
                self.add_cube(
                    Side_Length,
                    X_Coordinate,
                    Y_Coordinate,
                    Z_Coordinate,
                    R_v,
                    G_v,
                    B_v,
                    A_v,
                )
            
            case "Sphere":
                self.add_sphere(
                    Side_Length,
                    X_Coordinate,
                    Y_Coordinate,
                    Z_Coordinate,
                    R_v,
                    G_v,
                    B_v,
                    A_v,
                )

    def add_triangle(
        self,
        Side_Length: float,
        X_Coordinate: float,
        Y_Coordinate: float,
        Z_Coordinate: float,
        R_v: float,
        G_v: float,
        B_v: float,
        A_v: float,
    ) -> None:
        # fmt:off
        Vertices: NDArray[np.float32] = np.array([
            X_Coordinate - (Side_Length / 2), Y_Coordinate - (np.power(3,1 / 2) / 6 * Side_Length), Z_Coordinate, R_v, G_v, B_v, A_v,
            X_Coordinate + (Side_Length / 2), Y_Coordinate - (np.power(3,1 / 2) / 6 * Side_Length), Z_Coordinate, R_v, G_v, B_v, A_v,
            X_Coordinate, Y_Coordinate + (np.power(3,1 / 2) / 3 * Side_Length), Z_Coordinate, R_v, G_v, B_v, A_v,
            ],dtype=np.float32,
        )
        Indices: NDArray[np.uint32] = np.array([0, 1, 2], dtype=np.uint32)
        # fmt: on

        ObjectData: dict[str, Union[int, NDArray[np.float32], NDArray[np.uint32]]] = {
            "Type": GL_TRIANGLES,
            "Vertices": Vertices,
            "Indices": Indices,
        }

        self.window.OpenGLWindow.analysis_data(ObjectData)

    def add_square(
        self,
        Side_Length: float,
        X_Coordinate: float,
        Y_Coordinate: float,
        Z_Coordinate: float,
        R_v: float,
        G_v: float,
        B_v: float,
        A_v: float,
    ) -> None:
        # fmt: off
        Vertices: NDArray[np.float32] = np.array([
            X_Coordinate - (Side_Length / 2), Y_Coordinate + (Side_Length / 2), Z_Coordinate, R_v, G_v, B_v, A_v,
            X_Coordinate + (Side_Length / 2), Y_Coordinate + (Side_Length / 2), Z_Coordinate, R_v, G_v, B_v, A_v,
            X_Coordinate + (Side_Length / 2), Y_Coordinate - (Side_Length / 2), Z_Coordinate, R_v, G_v, B_v, A_v,
            X_Coordinate - (Side_Length / 2), Y_Coordinate - (Side_Length / 2), Z_Coordinate, R_v, G_v, B_v, A_v,
            ], dtype=np.float32
        )
        Indices: NDArray[np.uint32] = np.array([0, 1, 2, 3], dtype=np.uint32)
        # fmt: on

        ObjectData: dict[str, Union[int, NDArray[np.float32], NDArray[np.uint32]]] = {
            "Type": GL_QUADS,
            "Vertices": Vertices,
            "Indices": Indices,
        }

        self.window.OpenGLWindow.analysis_data(ObjectData)

    def add_cube(
        self,
        Side_Length: float,
        X_Coordinate: float,
        Y_Coordinate: float,
        Z_Coordinate: float,
        R_v: float,
        G_v: float,
        B_v: float,
        A_v: float,
    ) -> None:
        #    v4----- v5
        #   /|      /|
        #  v0------v1|
        #  | |     | |
        #  | v7----|-v6
        #  |/      |/
        #  v3------v2
        # fmt: off
        Vertices: NDArray[np.float32] = np.array([
            X_Coordinate - (Side_Length / 2), Y_Coordinate + (Side_Length / 2), Z_Coordinate + (Side_Length / 2), R_v, G_v, B_v, A_v,   # v0
            X_Coordinate + (Side_Length / 2), Y_Coordinate + (Side_Length / 2), Z_Coordinate + (Side_Length / 2), R_v, G_v, B_v, A_v,   # v1
            X_Coordinate + (Side_Length / 2), Y_Coordinate - (Side_Length / 2), Z_Coordinate + (Side_Length / 2), R_v, G_v, B_v, A_v,   # v2
            X_Coordinate - (Side_Length / 2), Y_Coordinate - (Side_Length / 2), Z_Coordinate + (Side_Length / 2), R_v, G_v, B_v, A_v,   # v3
            X_Coordinate - (Side_Length / 2), Y_Coordinate + (Side_Length / 2), Z_Coordinate - (Side_Length / 2), R_v, G_v, B_v, A_v,   # v4
            X_Coordinate + (Side_Length / 2), Y_Coordinate + (Side_Length / 2), Z_Coordinate - (Side_Length / 2), R_v, G_v, B_v, A_v,   # v5
            X_Coordinate + (Side_Length / 2), Y_Coordinate - (Side_Length / 2), Z_Coordinate - (Side_Length / 2), R_v, G_v, B_v, A_v,   # v6
            X_Coordinate - (Side_Length / 2), Y_Coordinate - (Side_Length / 2), Z_Coordinate - (Side_Length / 2), R_v, G_v, B_v, A_v,   # v7
        ], dtype=np.float32)
        Indices: NDArray[np.uint32] = np.array([
            0, 1, 2, 3, # v0-v1-v2-v3
            4, 5, 1, 0, # v4-v5-v1-v0
            3, 2, 6, 7, # v3-v2-v6-v7
            5, 4, 7, 6, # v5-v4-v7-v6
            1, 5, 6, 2, # v1-v5-v6-v2
            4, 0, 3, 7  # v4-v0-v3-v7
        ], dtype=np.uint32)
        # fmt: on
        ObjectData: dict[str, Union[int, NDArray[np.float32], NDArray[np.uint32]]] = {
            "Type": GL_QUADS,
            "Vertices": Vertices,
            "Indices": Indices,
        }

        self.window.OpenGLWindow.analysis_data(ObjectData)

    def add_sphere(
        self,
        Radius: float,
        X_Coordinate: float,
        Y_Coordinate: float,
        Z_Coordinate: float,
        R_v: float,
        G_v: float,
        B_v: float,
        A_v: float,
        Rings: int = 1000,
        Sectors: int = 1000,
    ) -> None:
        Vertices, Indices = Sphere_utils.generate_sphere(
            Radius,
            X_Coordinate,
            Y_Coordinate,
            Z_Coordinate,
            Rings,
            Sectors,
            R_v,
            G_v,
            B_v,
            A_v,
        )
        ObjectData: dict[str, Union[int, NDArray[np.float32], NDArray[np.uint32]]] = {
            "Type": GL_TRIANGLES,
            "Vertices": Vertices,
            "Indices": Indices,
        }

        self.window.OpenGLWindow.analysis_data(ObjectData)

    def delete_object(self) -> None:
        SelectedIndex = self.window.ObjectListView.currentIndex()
        if SelectedIndex.isValid():
            self.window.ObjectList.delete_selected(SelectedIndex)
        self.window.update()


class ObjectDataType(TypedDict):
    Type: str
    Side_Length: float
    X_Coordinate: float
    Y_Coordinate: float
    Z_Coordinate: float
    R: float
    G: float
    B: float
    A: float


class AddObjectDialog(QDialog):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Object")
        self.resize(600, 500)
        self.r: float = 0.0
        self.g: float = 0.0
        self.b: float = 0.0
        self.a: float = 0.0
        self.init_ui()

    def init_ui(self):
        Mainwidget = QWidget()
        MainLayout = QHBoxLayout(Mainwidget)

        self.SelectType = QComboBox()
        # fmt: off
        self.SelectType.addItems(["Equilateral triangle", "Square", "Cube", "Sphere"])
        # fmt: on

        self.Side_Length = QDoubleSpinBox()
        self.Side_Length.setRange(0, 100)
        self.Side_Length.setValue(5)
        self.Side_Length.setDecimals(2)

        self.X_Coordinate = QDoubleSpinBox()
        self.X_Coordinate.setRange(-100, 100)
        self.X_Coordinate.setValue(0)
        self.X_Coordinate.setDecimals(2)

        self.Y_Coordinate = QDoubleSpinBox()
        self.Y_Coordinate.setRange(-100, 100)
        self.Y_Coordinate.setValue(0)
        self.Y_Coordinate.setDecimals(2)

        self.Z_Coordinate = QDoubleSpinBox()
        self.Z_Coordinate.setRange(-100, 100)
        self.Z_Coordinate.setValue(0)
        self.Z_Coordinate.setDecimals(2)

        self.ColorSelector = QPushButton("Select Color")
        self.ColorResult = QLabel(
            f"RGBA: r={self.r}, g={self.g}, b={self.b}, a={self.a}"
        )

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
        MainLayout.addWidget(self.Accept_Reject)

        self.setLayout(MainLayout)

        self.ColorSelector.clicked.connect(self.show_color_dialog)
        self.Accept_Reject.accepted.connect(self.accept)
        self.Accept_Reject.rejected.connect(self.reject)

    def get_data(self) -> ObjectDataType:
        return {
            "Type": self.SelectType.currentText(),
            "Side_Length": self.Side_Length.value(),
            "X_Coordinate": self.X_Coordinate.value(),
            "Y_Coordinate": self.Y_Coordinate.value(),
            "Z_Coordinate": self.Z_Coordinate.value(),
            "R": self.r,
            "G": self.g,
            "B": self.b,
            "A": self.a,
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
