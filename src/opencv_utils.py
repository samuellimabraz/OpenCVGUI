import cv2
import numpy as np

from hand_tracker import HandTracker
from face_mesh_tracker import FaceMeshTracker


class OpenCVUtils:

    def __init__(self) -> None:
        self.hand_tracker = HandTracker(
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.face_mesh_tracker = FaceMeshTracker(
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def detect_faces(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Detect a face in the frame with the face mesh tracker of mediapipe

        :param frame: The frame to detect the face
        :param draw: If the output should be drawn
        """
        return self.face_mesh_tracker.detect(frame, draw=draw)

    def detect_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Detect a hand in the frame with the hand tracker of mediapipe

        :param frame: The frame to detect the hand
        :param draw: If the output should be drawn
        """
        result = self.hand_tracker.detect(frame, draw=draw)
        return result

    def apply_color_filter(
        self, frame: np.ndarray, lower_bound: list, upper_bound: list
    ) -> np.ndarray:
        """
        Apply a color filter to the frame

        :param frame: The frame to apply the filter
        :param lower_bound: The lower bound of the color filter in HSV
        :param upper_bound: The upper bound of the color filter in HSV
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([lower_bound[0], lower_bound[1], lower_bound[2]])
        upper_bound = np.array([upper_bound[0], upper_bound[1], upper_bound[2]])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def apply_edge_detection(
        self, frame: np.ndarray, lower_canny: int = 100, upper_canny: int = 200
    ) -> np.ndarray:
        """
        Apply a edge detection to the frame

        :param frame: The frame to apply the filter
        :param lower_canny: The lower bound of the canny edge detection
        :param upper_canny: The upper bound of the canny edge detection
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, lower_canny, upper_canny)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def apply_contour_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply a contour detection to the frame

        :param frame: The frame to apply the filter
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        return frame

    def blur_image(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply a blur to the image

        :param image: The image to apply the blur
        :param kernel_size: The kernel size of the blur
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def rotate_image(self, image: np.ndarray, angle: int = 0) -> np.ndarray:
        """
        Rotate the image

        :param image: The image to rotate
        :param angle: The angle to rotate the image
        """
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def resize_image(
        self, image: np.ndarray, width: int = None, height: int = None
    ) -> np.ndarray:
        """
        Resize the image

        :param image: The image to resize
        :param width: The width of the new image
        :param height: The height of the new image
        """
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def pencil_sketch(
        self,
        image: np.ndarray,
        sigma_s: int = 60,
        sigma_r: float = 0.07,
        shade_factor: float = 0.05,
    ) -> np.ndarray:
        # Converte para sketch preto e branco
        gray, sketch = cv2.pencilSketch(
            image, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor
        )
        return sketch

    def stylization(
        self, image: np.ndarray, sigma_s: int = 60, sigma_r: float = 0.45
    ) -> np.ndarray:
        # Efeito de pintura estilizada
        return cv2.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)

    def cartoonify(self, image: np.ndarray) -> np.ndarray:
        # Cartoon: detecta bordas e aplica quantização de cores
        # 1) Detecção de bordas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
        )
        # 2) Redução de cores
        data = np.float32(image).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, label, center = cv2.kmeans(
            data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        quant = center[label.flatten()].reshape(image.shape)
        # Combina bordas e quantização
        cartoon = cv2.bitwise_and(quant, quant, mask=edges)
        return cartoon

    def color_quantization(self, image: np.ndarray, k: int = 8) -> np.ndarray:
        # Reduz o número de cores via k-means
        data = np.float32(image).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, label, center = cv2.kmeans(
            data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        quant = center[label.flatten()].reshape(image.shape)
        return quant

    def equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])
        merged = cv2.merge(channels)
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2),
            cv2.COLOR_GRAY2BGR)

    def morphology(self, image: np.ndarray, op: str = 'erode', ksize: int = 5) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        ops = {
            'erode': cv2.erode,
            'dilate': cv2.dilate,
            'open': cv2.morphologyEx,
            'close': cv2.morphologyEx
        }
        if op in ['open', 'close']:
            flag = cv2.MORPH_OPEN if op == 'open' else cv2.MORPH_CLOSE
            return ops[op](image, flag, kernel)
        return ops[op](image, kernel)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def hough_lines(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:,0]:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return image

    def hough_circles(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=50, param1=50, param2=30,
                                   minRadius=5, maxRadius=100)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0, :]:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        return image

    def optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray, image: np.ndarray) -> np.ndarray:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros_like(image)
        hsv[...,1] = 255
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)