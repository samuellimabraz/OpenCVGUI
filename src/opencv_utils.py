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
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh_tracker = FaceMeshTracker(
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def detect_faces(self, frame, draw=True):
        return self.face_mesh_tracker.detect(frame, draw=draw)

    def detect_hands(self, frame, draw=False):
        return self.hand_tracker.detect(frame, draw=draw)

    def detect_hands_cvzone(self, frame, draw=True):
        _, img = self.hand_detector.findHands(frame, draw=draw, flipType=True)
        return img

    def apply_color_filter(self, frame, lower_bound, upper_bound):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([lower_bound[0], lower_bound[1], lower_bound[2]])
        upper_bound = np.array([upper_bound[0], upper_bound[1], upper_bound[2]])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return cv2.bitwise_and(frame, frame, mask=mask)

    # Função para aplicar a detecção de bordas
    def apply_edge_detection(self, frame, lower_canny, upper_canny):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, lower_canny, upper_canny)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Função para aplicar a detecção de contornos
    def apply_contour_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        return frame

    def blur_image(self, image, kernel_size=1):
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def rotate_image(self, image, angle=0):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def resize_image(self, image, width=None, height=None):
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
