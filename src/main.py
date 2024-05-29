import cv2
import numpy as np
import time

from tkinter import *
from tkinter import ttk

from PIL import Image, ImageTk

from opencv_utils import OpenCVUtils


class MainWindow:
    def __init__(self, root: Tk) -> None:
        self.root = root

        self.font = ("Arial", 12, "bold")
        self.font_small = ("Arial", 10, "bold")

        self.colors = {
            "yellow": "#FDCE01",
            "black": "#1E1E1E",
            "white": "#FEFEFE",
        }

        self.congig_interface()

        self.root.bind("<q>", self.close_application)

        self.functions = []
        self.aplication = OpenCVUtils()
        self.fps_avg_frame_count = 30

        self.COUNTER, self.FPS = 0, 0
        self.START_TIME = time.time()

    def close_application(self, event) -> None:
        """
        Close the application

        :param event: The event that triggered the function
        """
        # Libera a webcam e destrói todas as janelas do OpenCV
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def congig_interface(self) -> None:
        self.root.geometry("1500x1000")
        self.root.title("OpenCV + Tkinter")
        self.root.config(bg=self.colors["black"])

        self.paned_window = PanedWindow(self.root, orient=HORIZONTAL)
        self.paned_window.pack(fill=BOTH, expand=1)

        # Cria a barra lateral com os sliders
        self.sidebar = Frame(
            self.paned_window,
            width=700,
            bg=self.colors["black"],
            background=self.colors["black"],
            padx=10,
            pady=10,
        )
        self.paned_window.add(self.sidebar)

        # Cria as trackbars
        self.color_filter_var = IntVar()
        self.color_filter_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.apply_color_filter, self.color_filter_var
            ),
        )
        Checkbutton(
            self.sidebar,
            text="Color Filter",
            variable=self.color_filter_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        self.lower_hue = Scale(
            self.sidebar,
            from_=0,
            to=180,
            orient=HORIZONTAL,
            label="Lower Hue",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.lower_hue.pack(anchor="center")
        self.upper_hue = Scale(
            self.sidebar,
            from_=0,
            to=180,
            orient=HORIZONTAL,
            label="Upper Hue",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.upper_hue.pack(anchor="center")

        self.lower_saturation = Scale(
            self.sidebar,
            from_=0,
            to=255,
            orient=HORIZONTAL,
            label="Lower Sat",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.lower_saturation.pack(anchor="center")
        self.upper_saturation = Scale(
            self.sidebar,
            from_=0,
            to=255,
            orient=HORIZONTAL,
            label="Upper Sat",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.upper_saturation.pack(anchor="center")

        self.lower_value = Scale(
            self.sidebar,
            from_=0,
            to=255,
            orient=HORIZONTAL,
            label="Lower Value",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.lower_value.pack(anchor="center")
        self.upper_value = Scale(
            self.sidebar,
            from_=0,
            to=255,
            orient=HORIZONTAL,
            label="Upper Value",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.upper_value.pack(anchor="center")

        ttk.Separator(self.sidebar, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.canny_var = IntVar()
        self.canny_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.apply_edge_detection, self.canny_var
            ),
        )
        Checkbutton(
            self.sidebar,
            text="Canny",
            variable=self.canny_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        self.lower_canny = Scale(
            self.sidebar,
            from_=0,
            to=255,
            orient=HORIZONTAL,
            label="Lower Canny",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.lower_canny.pack(anchor="center")
        self.upper_canny = Scale(
            self.sidebar,
            from_=0,
            to=255,
            orient=HORIZONTAL,
            label="Upper Canny",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.upper_canny.pack(anchor="center")

        ttk.Separator(self.sidebar, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.blur_var = IntVar()
        self.blur_var.trace_add(
            "write",
            lambda *args: self.add_function(self.aplication.blur_image, self.blur_var),
        )
        Checkbutton(
            self.sidebar,
            text="Blur",
            variable=self.blur_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack(anchor="center")

        self.blur = Scale(
            self.sidebar,
            from_=1,
            to=15,
            orient=HORIZONTAL,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.blur.pack(anchor="center")

        ttk.Separator(self.sidebar, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.rotation_var = IntVar()
        self.rotation_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.rotate_image, self.rotation_var
            ),
        )
        Checkbutton(
            self.sidebar,
            text="Rotation",
            variable=self.rotation_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack(anchor="center")

        self.rotation_angle = Scale(
            self.sidebar,
            from_=0,
            to=360,
            orient=HORIZONTAL,
            label="Rotation Angle",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.rotation_angle.pack(anchor="center")

        ttk.Separator(self.sidebar, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.resize_var = IntVar()
        self.resize_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.resize_image, self.resize_var
            ),
        )
        Checkbutton(
            self.sidebar,
            text="Resize",
            variable=self.resize_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        Label(
            self.sidebar,
            text="Height",
            bg=self.colors["black"],
            fg=self.colors["white"],
        ).pack()
        self.height = Scale(
            self.sidebar,
            from_=100,
            to=1080,
            orient=HORIZONTAL,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.height.pack(anchor="center")
        self.width = Scale(
            self.sidebar,
            from_=100,
            to=1920,
            orient=HORIZONTAL,
            label="Width",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.width.pack(anchor="center")

        ttk.Separator(self.sidebar, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.contour_var = IntVar()
        self.contour_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.apply_contour_detection, self.contour_var
            ),
        )
        Checkbutton(
            self.sidebar,
            text="Contour",
            variable=self.contour_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(self.sidebar, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.hand_tracker_var = IntVar()
        self.hand_tracker_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.detect_hands, self.hand_tracker_var
            ),
        )
        Checkbutton(
            self.sidebar,
            text="Hand Tracker",
            variable=self.hand_tracker_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(self.sidebar, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.face_tracker_var = IntVar()
        self.face_tracker_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.detect_faces, self.face_tracker_var
            ),
        )
        Checkbutton(
            self.sidebar,
            text="Face Tracker",
            variable=self.face_tracker_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        # Cria o label para exibir a imagem
        self.image_label = Label(self.paned_window, bg=self.colors["black"])
        self.paned_window.add(self.image_label)

    def add_function(self, function: callable, var: IntVar) -> None:
        """
        Add or remove a function from the list of functions to be applied to the image

        :param function: The function to be added or removed
        :param var: The variable that controls the function
        """
        if var.get() == 1:
            self.functions.append(function)
        else:
            self.functions.remove(function)

    def process_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Process the image with the functions selected by the user

        :param frame: The image to be processed
        :return: The processed image
        """
        function_dict = {
            self.aplication.apply_color_filter: [
                (
                    self.lower_hue.get(),
                    self.lower_saturation.get(),
                    self.lower_value.get(),
                ),
                (
                    self.upper_hue.get(),
                    self.upper_saturation.get(),
                    self.upper_value.get(),
                ),
            ],
            self.aplication.apply_edge_detection: [
                self.lower_canny.get(),
                self.upper_canny.get(),
            ],
            self.aplication.blur_image: [self.blur.get()],
            self.aplication.rotate_image: [self.rotation_angle.get()],
            self.aplication.resize_image: [self.width.get(), self.height.get()],
        }

        for function in self.functions:
            args = function_dict.get(function, [])
            frame = function(frame, *args)

        return frame

    def run(self) -> None:
        """
        Run the main loop of the tkinter application
        """
        # Abre a webcam
        self.cap = cv2.VideoCapture(0)
        self.START_TIME = time.time()
        while True:
            # Lê um frame da webcam
            ret, frame = self.cap.read()
            if not ret:
                break

            # Aplica as funções do OpenCV
            frame = self.process_image(frame)

            output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.COUNTER % self.fps_avg_frame_count == 0:
                self.FPS = self.fps_avg_frame_count / (time.time() - self.START_TIME)
                self.START_TIME = time.time()
            self.COUNTER += 1

            # Show the FPS
            fps_text = "FPS = {:.1f}".format(self.FPS)

            cv2.putText(
                output,
                fps_text,
                (24, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            # Converte a imagem NumPy para uma imagem PIL
            pil_image = Image.fromarray(output)

            # Converte a imagem PIL para uma imagem Tkinter
            tk_image = ImageTk.PhotoImage(pil_image)

            # Exibe a imagem no label
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

            # Atualiza a janela tkinter
            self.root.update()

            cv2.waitKey(1)


def main():
    # Cria a janela principal
    root = Tk()
    main_window = MainWindow(root)
    main_window.run()


if __name__ == "__main__":
    main()
