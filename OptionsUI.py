import dearpygui.dearpygui as dpg
import numpy as np


class createUI:
    def __init__(self, RenderQueue):
        self.RenderQueue = RenderQueue
        self.init_ui()

    def init_ui(self):
        dpg.create_context()
        dpg.show_documentation()
        dpg.create_viewport(title="test", width=1000, height=600)

        with dpg.window(
            label="Tune Parameters",
            tag="parameters",
            width=200,
            no_move=True,
            no_resize=True,
        ):
            dpg.add_text("Parameters")
            dpg.add_slider_float(label="test2", default_value=0.5, max_value=1)
            dpg.add_slider_float(label="test2", default_value=0.5, max_value=1)
            dpg.add_slider_float(label="test2", default_value=0.5, max_value=1)
            dpg.add_slider_float(label="test2", default_value=0.5, max_value=1)
            dpg.add_slider_float(label="test2", default_value=0.5, max_value=1)
            dpg.add_slider_float(label="test2", default_value=0.5, max_value=1)

        with dpg.window(
            label="Simulator", tag="simulator", no_move=True, no_resize=True
        ):
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=800,
                    height=600,
                    default_value=np.zeros((600, 800), dtype=np.float32),
                    format=dpg.mvFormat_Float_rgba,
                    tag="opengl",
                )

            dpg.add_image("opengl")

        with dpg.window(
            label="Options", tag="options", width=200, no_move=True, no_resize=True
        ):
            dpg.add_button(label="test")
            dpg.add_button(label="test")
            dpg.add_button(label="test")
            dpg.add_button(label="test")
            dpg.add_button(label="test")
            dpg.add_button(label="test")
            dpg.add_button(label="test")

        dpg.set_viewport_resize_callback(self.Resize_Callback)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.Resize_Callback()
        while dpg.is_dearpygui_running():
            self.update_opengl()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def Resize_Callback(self):
        width = dpg.get_viewport_width()
        height = dpg.get_viewport_height()

        dpg.configure_item("parameters", pos=(0, 0), height=height)
        dpg.configure_item("options", pos=(width - 200, 0), height=height)
        dpg.configure_item("simulator", pos=(200, 0), width=width - 400, height=height)

    def update_opengl(self):
        if not self.RenderQueue.empty():
            frame = self.RenderQueue.get()
            texture_data = (frame / 255.0).astype(np.float32)
            dpg.set_value("opengl", texture_data.reshape(-1))

