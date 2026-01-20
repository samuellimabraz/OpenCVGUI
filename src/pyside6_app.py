#!/usr/bin/env python3
import sys
import time
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QCheckBox,
    QComboBox,
    QScrollArea,
    QFrame,
    QSplitter,
    QRadioButton,
    QButtonGroup,
    QPushButton,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QFont

from opencv_utils import OpenCVUtils


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._is_collapsed = False
        self._content_height = 0

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.toggle_button = QPushButton(f"▼ {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setStyleSheet(
            """
            QPushButton {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: none;
                padding: 10px 15px;
                text-align: left;
                font-weight: bold;
                font-size: 13px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:checked {
                background-color: #2d2d2d;
            }
        """
        )
        self.toggle_button.clicked.connect(self._toggle)

        self.content_area = QFrame()
        self.content_area.setStyleSheet(
            """
            QFrame {
                background-color: transparent;
                border: none;
            }
        """
        )
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(8)

        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)

    def _toggle(self):
        self._is_collapsed = not self.toggle_button.isChecked()
        self.content_area.setVisible(not self._is_collapsed)
        arrow = "▶" if self._is_collapsed else "▼"
        title = self.toggle_button.text()[2:]
        self.toggle_button.setText(f"{arrow} {title}")

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


class StyledSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(
        self, label: str, min_val: int, max_val: int, default: int = 0, parent=None
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.label = QLabel(label)
        self.label.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        self.value_label = QLabel(str(default))
        self.value_label.setStyleSheet(
            "color: #00d4aa; font-size: 12px; font-weight: bold;"
        )
        self.value_label.setAlignment(Qt.AlignRight)
        header.addWidget(self.label)
        header.addWidget(self.value_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default)
        self.slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                background: #3d3d3d;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00d4aa;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #00ffcc;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d4aa, stop:1 #00a88a);
                border-radius: 3px;
            }
        """
        )
        self.slider.valueChanged.connect(self._on_value_changed)

        layout.addLayout(header)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value):
        self.value_label.setText(str(value))
        self.valueChanged.emit(value)

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        self.slider.setValue(value)


class StyledCheckBox(QCheckBox):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            """
            QCheckBox {
                color: #e0e0e0;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #4a4a4a;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background-color: #00d4aa;
                border-color: #00d4aa;
            }
            QCheckBox::indicator:hover {
                border-color: #00d4aa;
            }
        """
        )


class StyledComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            """
            QComboBox {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #00d4aa;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #00d4aa;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: #e0e0e0;
                selection-background-color: #00d4aa;
                selection-color: #1a1a1a;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
            }
        """
        )


class VideoDisplay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(
            """
            QLabel {
                background-color: #0a0a0a;
                border: 2px solid #2d2d2d;
                border-radius: 12px;
            }
        """
        )
        self.setScaledContents(False)


class OpenCVExplorerQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Explorer - Qt6")
        self.setMinimumSize(1400, 900)

        self.cv_utils = OpenCVUtils()
        self.functions = []
        self.prev_gray = None

        self.fps_counter = 0
        self.fps = 0.0
        self.fps_start_time = time.time()
        self.fps_avg_count = 30

        self.cap = None

        self._setup_ui()
        self._setup_camera()
        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #1a1a1a;
                color: #e0e0e0;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #1a1a1a;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #3d3d3d;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #00d4aa;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QSplitter::handle {
                background-color: #2d2d2d;
                width: 3px;
            }
        """
        )

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        splitter = QSplitter(Qt.Horizontal)

        # Control panel
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)

        # Video display area
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Live Preview")
        header.setStyleSheet(
            """
            QLabel {
                color: #00d4aa;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
            }
        """
        )
        header.setAlignment(Qt.AlignCenter)

        self.video_display = VideoDisplay()

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet(
            """
            QLabel {
                color: #808080;
                font-size: 12px;
                padding: 5px;
            }
        """
        )
        self.fps_label.setAlignment(Qt.AlignCenter)

        video_layout.addWidget(header)
        video_layout.addWidget(self.video_display, 1)
        video_layout.addWidget(self.fps_label)

        splitter.addWidget(video_container)
        splitter.setSizes([380, 1000])

        main_layout.addWidget(splitter)

    def _create_control_panel(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(360)
        scroll_area.setMaximumWidth(420)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        title = QLabel("Filters & Effects")
        title.setStyleSheet(
            """
            QLabel {
                color: #00d4aa;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 10px;
            }
        """
        )
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Color Filter Section
        self._create_color_filter_section(layout)

        # Edge Detection Section
        self._create_edge_detection_section(layout)

        # Blur Section
        self._create_blur_section(layout)

        # Transform Section
        self._create_transform_section(layout)

        # Morphology Section
        self._create_morphology_section(layout)

        # Effects Section
        self._create_effects_section(layout)

        # Detection Section
        self._create_detection_section(layout)

        # ArUco Section
        self._create_aruco_section(layout)

        layout.addStretch()
        scroll_area.setWidget(container)
        return scroll_area

    def _create_color_filter_section(self, parent_layout):
        section = CollapsibleSection("🎨 Color Filter")

        self.color_filter_cb = StyledCheckBox("Enable Color Filter")
        self.color_filter_cb.stateChanged.connect(
            lambda: self._toggle_function(
                self.cv_utils.apply_color_filter, self.color_filter_cb
            )
        )
        section.add_widget(self.color_filter_cb)

        self.lower_hue = StyledSlider("Lower Hue", 0, 180, 0)
        self.upper_hue = StyledSlider("Upper Hue", 0, 180, 180)
        self.lower_sat = StyledSlider("Lower Saturation", 0, 255, 0)
        self.upper_sat = StyledSlider("Upper Saturation", 0, 255, 255)
        self.lower_val = StyledSlider("Lower Value", 0, 255, 0)
        self.upper_val = StyledSlider("Upper Value", 0, 255, 255)

        section.add_widget(self.lower_hue)
        section.add_widget(self.upper_hue)
        section.add_widget(self.lower_sat)
        section.add_widget(self.upper_sat)
        section.add_widget(self.lower_val)
        section.add_widget(self.upper_val)

        parent_layout.addWidget(section)

    def _create_edge_detection_section(self, parent_layout):
        section = CollapsibleSection("🔍 Edge Detection")

        self.canny_cb = StyledCheckBox("Enable Canny Edge")
        self.canny_cb.stateChanged.connect(
            lambda: self._toggle_function(
                self.cv_utils.apply_edge_detection, self.canny_cb
            )
        )
        section.add_widget(self.canny_cb)

        self.lower_canny = StyledSlider("Lower Threshold", 0, 255, 100)
        self.upper_canny = StyledSlider("Upper Threshold", 0, 255, 200)

        section.add_widget(self.lower_canny)
        section.add_widget(self.upper_canny)

        self.contour_cb = StyledCheckBox("Enable Contour Detection")
        self.contour_cb.stateChanged.connect(
            lambda: self._toggle_function(
                self.cv_utils.apply_contour_detection, self.contour_cb
            )
        )
        section.add_widget(self.contour_cb)

        parent_layout.addWidget(section)

    def _create_blur_section(self, parent_layout):
        section = CollapsibleSection("🌫️ Blur & Sharpen")

        self.blur_cb = StyledCheckBox("Enable Gaussian Blur")
        self.blur_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.blur_image, self.blur_cb)
        )
        section.add_widget(self.blur_cb)

        self.blur_kernel = StyledSlider("Kernel Size", 1, 31, 5)
        section.add_widget(self.blur_kernel)

        self.sharpen_cb = StyledCheckBox("Enable Sharpen")
        self.sharpen_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.sharpen, self.sharpen_cb)
        )
        section.add_widget(self.sharpen_cb)

        parent_layout.addWidget(section)

    def _create_transform_section(self, parent_layout):
        section = CollapsibleSection("🔄 Transformations")

        self.rotation_cb = StyledCheckBox("Enable Rotation")
        self.rotation_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.rotate_image, self.rotation_cb)
        )
        section.add_widget(self.rotation_cb)

        self.rotation_angle = StyledSlider("Angle", 0, 360, 0)
        section.add_widget(self.rotation_angle)

        self.resize_cb = StyledCheckBox("Enable Resize")
        self.resize_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.resize_image, self.resize_cb)
        )
        section.add_widget(self.resize_cb)

        self.resize_width = StyledSlider("Width", 100, 1920, 640)
        self.resize_height = StyledSlider("Height", 100, 1080, 480)
        section.add_widget(self.resize_width)
        section.add_widget(self.resize_height)

        parent_layout.addWidget(section)

    def _create_morphology_section(self, parent_layout):
        section = CollapsibleSection("⚙️ Morphological Ops")

        self.morphology_cb = StyledCheckBox("Enable Morphology")
        self.morphology_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.morphology, self.morphology_cb)
        )
        section.add_widget(self.morphology_cb)

        op_label = QLabel("Operation:")
        op_label.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        section.add_widget(op_label)

        self.morph_op_group = QButtonGroup(self)
        self.morph_op = "erode"

        op_container = QWidget()
        op_layout = QHBoxLayout(op_container)
        op_layout.setContentsMargins(0, 0, 0, 0)
        op_layout.setSpacing(10)

        for op in ["erode", "dilate", "open", "close"]:
            rb = QRadioButton(op.capitalize())
            rb.setStyleSheet(
                """
                QRadioButton {
                    color: #b0b0b0;
                    font-size: 11px;
                }
                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                }
                QRadioButton::indicator:checked {
                    background-color: #00d4aa;
                    border-radius: 8px;
                }
                QRadioButton::indicator:unchecked {
                    background-color: #3d3d3d;
                    border-radius: 8px;
                }
            """
            )
            if op == "erode":
                rb.setChecked(True)
            rb.toggled.connect(
                lambda checked, o=op: self._set_morph_op(o) if checked else None
            )
            self.morph_op_group.addButton(rb)
            op_layout.addWidget(rb)

        section.add_widget(op_container)

        self.morph_kernel = StyledSlider("Kernel Size", 1, 31, 5)
        section.add_widget(self.morph_kernel)

        self.adaptive_thresh_cb = StyledCheckBox("Adaptive Threshold")
        self.adaptive_thresh_cb.stateChanged.connect(
            lambda: self._toggle_function(
                self.cv_utils.adaptive_threshold, self.adaptive_thresh_cb
            )
        )
        section.add_widget(self.adaptive_thresh_cb)

        self.hist_eq_cb = StyledCheckBox("Histogram Equalization")
        self.hist_eq_cb.stateChanged.connect(
            lambda: self._toggle_function(
                self.cv_utils.equalize_histogram, self.hist_eq_cb
            )
        )
        section.add_widget(self.hist_eq_cb)

        parent_layout.addWidget(section)

    def _create_effects_section(self, parent_layout):
        section = CollapsibleSection("✨ Visual Effects")

        self.pencil_cb = StyledCheckBox("Pencil Sketch")
        self.pencil_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.pencil_sketch, self.pencil_cb)
        )
        section.add_widget(self.pencil_cb)

        self.stylize_cb = StyledCheckBox("Stylization")
        self.stylize_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.stylization, self.stylize_cb)
        )
        section.add_widget(self.stylize_cb)

        self.cartoon_cb = StyledCheckBox("Cartoonify")
        self.cartoon_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.cartoonify, self.cartoon_cb)
        )
        section.add_widget(self.cartoon_cb)

        self.quantize_cb = StyledCheckBox("Color Quantization")
        self.quantize_cb.stateChanged.connect(
            lambda: self._toggle_function(
                self.cv_utils.color_quantization, self.quantize_cb
            )
        )
        section.add_widget(self.quantize_cb)

        self.hough_lines_cb = StyledCheckBox("Hough Lines")
        self.hough_lines_cb.stateChanged.connect(
            lambda: self._toggle_function(
                self.cv_utils.hough_lines, self.hough_lines_cb
            )
        )
        section.add_widget(self.hough_lines_cb)

        self.hough_circles_cb = StyledCheckBox("Hough Circles")
        self.hough_circles_cb.stateChanged.connect(
            lambda: self._toggle_function(
                self.cv_utils.hough_circles, self.hough_circles_cb
            )
        )
        section.add_widget(self.hough_circles_cb)

        self.optical_flow_cb = StyledCheckBox("Optical Flow")
        self.optical_flow_cb.stateChanged.connect(
            lambda: self._toggle_function(
                self._process_optical_flow, self.optical_flow_cb
            )
        )
        section.add_widget(self.optical_flow_cb)

        parent_layout.addWidget(section)

    def _create_detection_section(self, parent_layout):
        section = CollapsibleSection("👋 AI Detection")

        self.hand_cb = StyledCheckBox("Hand Tracking")
        self.hand_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.detect_hands, self.hand_cb)
        )
        section.add_widget(self.hand_cb)

        self.face_cb = StyledCheckBox("Face Mesh")
        self.face_cb.stateChanged.connect(
            lambda: self._toggle_function(self.cv_utils.detect_faces, self.face_cb)
        )
        section.add_widget(self.face_cb)

        parent_layout.addWidget(section)

    def _create_aruco_section(self, parent_layout):
        section = CollapsibleSection("📐 ArUco Markers")

        self.aruco_cb = StyledCheckBox("Enable ArUco Detection")
        self.aruco_cb.stateChanged.connect(
            lambda: self._toggle_function(self._detect_aruco, self.aruco_cb)
        )
        section.add_widget(self.aruco_cb)

        dict_label = QLabel("Dictionary:")
        dict_label.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        section.add_widget(dict_label)

        self.aruco_dict_combo = StyledComboBox()
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
        self.aruco_dict_combo.addItems(aruco_dicts)
        self.aruco_dict_combo.setCurrentIndex(10)
        section.add_widget(self.aruco_dict_combo)

        parent_layout.addWidget(section)

    def _set_morph_op(self, op):
        self.morph_op = op

    def _toggle_function(self, func, checkbox):
        if checkbox.isChecked():
            if func not in self.functions:
                self.functions.append(func)
        else:
            if func in self.functions:
                self.functions.remove(func)

    def _process_optical_flow(self, frame: np.ndarray) -> np.ndarray:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is not None:
            frame = self.cv_utils.optical_flow(self.prev_gray, curr_gray, frame)
        self.prev_gray = curr_gray
        return frame

    def _detect_aruco(self, frame: np.ndarray) -> np.ndarray:
        return self.cv_utils.detect_aruco_markers(
            frame, dict_type=self.aruco_dict_combo.currentText()
        )

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        function_params = {
            self.cv_utils.apply_color_filter: [
                (
                    self.lower_hue.value(),
                    self.lower_sat.value(),
                    self.lower_val.value(),
                ),
                (
                    self.upper_hue.value(),
                    self.upper_sat.value(),
                    self.upper_val.value(),
                ),
            ],
            self.cv_utils.apply_edge_detection: [
                self.lower_canny.value(),
                self.upper_canny.value(),
            ],
            self.cv_utils.blur_image: [self.blur_kernel.value()],
            self.cv_utils.rotate_image: [self.rotation_angle.value()],
            self.cv_utils.resize_image: [
                self.resize_width.value(),
                self.resize_height.value(),
            ],
            self.cv_utils.morphology: [self.morph_op, self.morph_kernel.value()],
        }

        for func in self.functions:
            args = function_params.get(func, [])
            frame = func(frame, *args)

        return frame

    def _setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(16)

    def _update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = self._process_frame(frame)

        self.fps_counter += 1
        if self.fps_counter % self.fps_avg_count == 0:
            elapsed = time.time() - self.fps_start_time
            self.fps = self.fps_avg_count / elapsed if elapsed > 0 else 0
            self.fps_start_time = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.putText(
            rgb_frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 212, 170),
            2,
            cv2.LINE_AA,
        )

        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        display_size = self.video_display.size()
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_display.setPixmap(scaled_pixmap)
        self.fps_label.setText(f"FPS: {self.fps:.1f} | Resolution: {w}x{h}")

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = OpenCVExplorerQt()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
