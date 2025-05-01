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

        # For optical flow
        self.prev_gray = None

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

        # Create a scrollbar for the sidebar
        canvas = Canvas(self.sidebar, bg=self.colors["black"], highlightthickness=0)
        scrollbar = Scrollbar(self.sidebar, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(
            canvas,
            bg=self.colors["black"],
        )

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Cria as trackbars
        self.color_filter_var = IntVar()
        self.color_filter_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.apply_color_filter, self.color_filter_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Color Filter",
            variable=self.color_filter_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        self.lower_hue = Scale(
            scrollable_frame,
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
            scrollable_frame,
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
            scrollable_frame,
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
            scrollable_frame,
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
            scrollable_frame,
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
            scrollable_frame,
            from_=0,
            to=255,
            orient=HORIZONTAL,
            label="Upper Value",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.upper_value.pack(anchor="center")

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.canny_var = IntVar()
        self.canny_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.apply_edge_detection, self.canny_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Canny",
            variable=self.canny_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        self.lower_canny = Scale(
            scrollable_frame,
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
            scrollable_frame,
            from_=0,
            to=255,
            orient=HORIZONTAL,
            label="Upper Canny",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.upper_canny.pack(anchor="center")

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.blur_var = IntVar()
        self.blur_var.trace_add(
            "write",
            lambda *args: self.add_function(self.aplication.blur_image, self.blur_var),
        )
        Checkbutton(
            scrollable_frame,
            text="Blur",
            variable=self.blur_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack(anchor="center")

        self.blur = Scale(
            scrollable_frame,
            from_=1,
            to=15,
            orient=HORIZONTAL,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.blur.pack(anchor="center")

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.rotation_var = IntVar()
        self.rotation_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.rotate_image, self.rotation_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Rotation",
            variable=self.rotation_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack(anchor="center")

        self.rotation_angle = Scale(
            scrollable_frame,
            from_=0,
            to=360,
            orient=HORIZONTAL,
            label="Rotation Angle",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.rotation_angle.pack(anchor="center")

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.resize_var = IntVar()
        self.resize_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.resize_image, self.resize_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Resize",
            variable=self.resize_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        Label(
            scrollable_frame,
            text="Height",
            bg=self.colors["black"],
            fg=self.colors["white"],
        ).pack()
        self.height = Scale(
            scrollable_frame,
            from_=100,
            to=1080,
            orient=HORIZONTAL,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.height.pack(anchor="center")
        self.width = Scale(
            scrollable_frame,
            from_=100,
            to=1920,
            orient=HORIZONTAL,
            label="Width",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.width.pack(anchor="center")

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.contour_var = IntVar()
        self.contour_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.apply_contour_detection, self.contour_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Contour",
            variable=self.contour_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        # Add new OpenCV functions

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.hist_equal_var = IntVar()
        self.hist_equal_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.equalize_histogram, self.hist_equal_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Histogram Equalization",
            variable=self.hist_equal_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.adaptive_threshold_var = IntVar()
        self.adaptive_threshold_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.adaptive_threshold, self.adaptive_threshold_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Adaptive Threshold",
            variable=self.adaptive_threshold_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.morphology_var = IntVar()
        self.morphology_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.morphology, self.morphology_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Morphology",
            variable=self.morphology_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        # Morphology operation options
        self.morph_op_var = StringVar(value="erode")
        Label(
            scrollable_frame,
            text="Operation",
            bg=self.colors["black"],
            fg=self.colors["white"],
        ).pack()

        for op in ["erode", "dilate", "open", "close"]:
            Radiobutton(
                scrollable_frame,
                text=op.capitalize(),
                variable=self.morph_op_var,
                value=op,
                bg=self.colors["black"],
                fg=self.colors["white"],
                selectcolor=self.colors["black"],
                highlightbackground=self.colors["black"],
            ).pack(anchor="w")

        self.morph_kernel_size = Scale(
            scrollable_frame,
            from_=1,
            to=31,
            orient=HORIZONTAL,
            label="Kernel Size",
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
        )
        self.morph_kernel_size.set(5)
        self.morph_kernel_size.pack(anchor="center")

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.sharpen_var = IntVar()
        self.sharpen_var.trace_add(
            "write",
            lambda *args: self.add_function(self.aplication.sharpen, self.sharpen_var),
        )
        Checkbutton(
            scrollable_frame,
            text="Sharpen",
            variable=self.sharpen_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.hough_lines_var = IntVar()
        self.hough_lines_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.hough_lines, self.hough_lines_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Hough Lines",
            variable=self.hough_lines_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.optical_flow_var = IntVar()
        self.optical_flow_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.process_optical_flow, self.optical_flow_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Optical Flow",
            variable=self.optical_flow_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.pencil_sketch_var = IntVar()
        self.pencil_sketch_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.pencil_sketch, self.pencil_sketch_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Pencil Sketch",
            variable=self.pencil_sketch_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.color_quantization_var = IntVar()
        self.color_quantization_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.color_quantization, self.color_quantization_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Color Quantization",
            variable=self.color_quantization_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.hand_tracker_var = IntVar()
        self.hand_tracker_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.detect_hands, self.hand_tracker_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Hand Tracker",
            variable=self.hand_tracker_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        self.face_tracker_var = IntVar()
        self.face_tracker_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.aplication.detect_faces, self.face_tracker_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="Face Tracker",
            variable=self.face_tracker_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        ttk.Separator(scrollable_frame, orient=HORIZONTAL).pack(fill=X, padx=3, pady=3)

        # Add ArUco Marker Detector
        self.aruco_marker_var = IntVar()
        self.aruco_marker_var.trace_add(
            "write",
            lambda *args: self.add_function(
                self.detect_aruco_markers, self.aruco_marker_var
            ),
        )
        Checkbutton(
            scrollable_frame,
            text="ArUco Marker Detector",
            variable=self.aruco_marker_var,
            font=self.font,
            bg=self.colors["black"],
            fg=self.colors["white"],
            highlightbackground=self.colors["black"],
            selectcolor=self.colors["black"],
        ).pack()

        # ArUco dictionary selector
        Label(
            scrollable_frame,
            text="ArUco Dictionary",
            bg=self.colors["black"],
            fg=self.colors["white"],
        ).pack()

        self.aruco_dict_var = StringVar(value="DICT_6X6_250")
        aruco_dicts = [
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_4X4_250",
            "DICT_4X4_1000",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_5X5_250",
            "DICT_5X5_1000",
            "DICT_6X6_50",
            "DICT_6X6_100",
            "DICT_6X6_250",
            "DICT_6X6_1000",
            "DICT_7X7_50",
            "DICT_7X7_100",
            "DICT_7X7_250",
            "DICT_7X7_1000",
            "DICT_ARUCO_ORIGINAL",
        ]

        # Create a combobox for selecting dictionary
        aruco_dict_combo = ttk.Combobox(
            scrollable_frame,
            textvariable=self.aruco_dict_var,
            values=aruco_dicts,
            state="readonly",
            width=20,
        )
        aruco_dict_combo.pack(pady=5)
        aruco_dict_combo.current(10)  # Default to DICT_6X6_250

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

    def detect_aruco_markers(self, frame: np.ndarray) -> np.ndarray:
        """
        Wrapper for ArUco marker detection to pass the dictionary type parameter

        :param frame: The frame to detect ArUco markers
        :return: The frame with detected ArUco markers
        """
        return self.aplication.detect_aruco_markers(
            frame, dict_type=self.aruco_dict_var.get()
        )

    def process_optical_flow(self, frame: np.ndarray) -> np.ndarray:
        """
        Special handler for optical flow which needs to track previous frames

        :param frame: The current frame
        :return: The processed frame with optical flow
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is not None:
            frame = self.aplication.optical_flow(self.prev_gray, curr_gray, frame)

        self.prev_gray = curr_gray
        return frame

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
            self.aplication.morphology: [
                self.morph_op_var.get(),
                self.morph_kernel_size.get(),
            ],
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
