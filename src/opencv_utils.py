import cv2
import numpy as np

from hand_tracker import HandTracker
from face_mesh_tracker import FaceMeshTracker

from cvzone.HandTrackingModule import HandDetector


class OpenCVUtils:

    def __init__(self) -> None:
        self.hand_detector = HandDetector(
            staticMode=False,
            maxHands=2,
            modelComplexity=1,
            detectionCon=0.5,
            minTrackCon=0.5,
        )
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

    def detect_hands_cvzone(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Detect a hand in the frame with the hand detector of cvzone

        :param frame: The frame to detect the hand
        :param draw: If the output should be drawn
        """
        _, img = self.hand_detector.findHands(frame, draw=draw, flipType=True)
        return img

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
