import os
import urllib.request
import sys

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import time
import numpy as np

# import autopy


class FaceMeshTracker:
    # face bounder indices
    FACE_OVAL = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]

    # lips indices for Landmarks
    LIPS = [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        42,
        183,
        78,
    ]
    LOWER_LIPS = [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
    ]
    UPPER_LIPS = [
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        415,
        310,
        311,
        312,
        13,
        82,
        81,
        42,
        183,
        78,
    ]
    # Left eyes indices
    LEFT_EYE = [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    LEFT_CENTER_EYE = [473]

    # right eyes indices
    RIGHT_EYE = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_CENTER_EYE = [468]

    def __init__(
        self,
        model: str = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize a FaceTracker instance.

        Args:
            model (str): The path to the model for face tracking.
            num_faces (int): Maximum number of faces to detect.
            min_face_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for successful face detection.
            min_face_presence_confidence (float): Minimum confidence value ([0.0, 1.0]) for presence of a face to be tracked.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for successful face landmark tracking.
        """
        self.model = model

        if self.model == None:
            self.model = self.download_model()

        if self.model == None:
            self.model = self.download_model()

        self.detector = self.initialize_detector(
            num_faces,
            min_face_detection_confidence,
            min_face_presence_confidence,
            min_tracking_confidence,
        )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.DETECTION_RESULT = None

    def save_result(
        self,
        result: vision.FaceLandmarkerResult,
        unused_output_image,
        timestamp_ms: int,
        fps: bool = False,
    ):
        """
        Saves the result of the face detection.

        Args:
            result (vision.FaceLandmarkerResult): Result of the face detection.
            unused_output_image (mp.Image): Unused.
            timestamp_ms (int): Timestamp of the detection.

        Returns:
            None
        """
        self.DETECTION_RESULT = result

    def initialize_detector(
        self,
        num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float,
        min_tracking_confidence: float,
    ):
        """
        Initializes the FaceLandmarker instance.

        Args:
            num_faces (int): Maximum number of faces to detect.
            min_face_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.
            min_face_presence_confidence (float): Minimum confidence value ([0.0, 1.0]) for the presence of a face for the face landmarks to be considered tracked successfully.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for the face landmarks to be considered tracked successfully.

        Returns:
            vision.FaceLandmarker: FaceLandmarker instance.
        """
        base_options = python.BaseOptions(model_asset_path=self.model)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            result_callback=self.save_result,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def draw_landmarks(
        self,
        image: np.ndarray,
        text_color: tuple = (0, 0, 0),
        font_size: int = 1,
        font_thickness: int = 1,
    ) -> np.ndarray:
        """
        Draws the face landmarks on the image.

        Args:
            image (numpy.ndarray): Image on which to draw the landmarks.
            text_color (tuple, optional): Color of the text. Defaults to (0, 0, 0).
            font_size (int, optional): Size of the font. Defaults to 1.
            font_thickness (int, optional): Thickness of the font. Defaults to 1.

        Returns:
            numpy.ndarray: Image with the landmarks drawn.
        """

        if self.DETECTION_RESULT:
            # Draw landmarks.
            for face_landmarks in self.DETECTION_RESULT.face_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in face_landmarks
                    ]
                )
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks_proto,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        return image

    def draw_landmark_circles(
        self,
        image: np.ndarray,
        landmark_indices: list,
        circle_radius: int = 1,
        circle_color: tuple = (0, 255, 0),
        circle_thickness: int = 1,
    ) -> np.ndarray:
        """
        Draws circles on the specified face landmarks on the image.

        Args:
            image (numpy.ndarray): Image on which to draw the landmarks.
            landmark_indices (list of int): Indices of the landmarks to draw.
            circle_radius (int, optional): Radius of the circles. Defaults to 1.
            circle_color (tuple, optional): Color of the circles. Defaults to (0, 255, 0).
            circle_thickness (int, optional): Thickness of the circles. Defaults to 1.

        Returns:
            numpy.ndarray: Image with the landmarks drawn.
        """
        if self.DETECTION_RESULT:
            # Draw landmarks.
            for face_landmarks in self.DETECTION_RESULT.face_landmarks:
                for i, landmark in enumerate(face_landmarks):
                    if i in landmark_indices:
                        # Convert the landmark position to image coordinates.
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(
                            image,
                            (x, y),
                            circle_radius,
                            circle_color,
                            circle_thickness,
                        )

        return image

    def detect(self, frame: np.ndarray, draw: bool = False) -> np.ndarray:
        """
        Detects the face landmarks in the frame.

        Args:
            frame (numpy.ndarray): Frame in which to detect the landmarks.
            draw (bool, optional): Whether to draw the landmarks on the frame. Defaults to False.

        Returns:
            numpy.ndarray: Frame with the landmarks drawn.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.detector.detect_async(mp_image, time.time_ns() // 1_000_000)
        return self.draw_landmarks(frame) if draw else frame

    def get_face_landmarks(self, face_idx: int = 0, idxs: list = None) -> list:
        """
        Returns the face landmarks.

        Args:
            face_idx (int, optional): Index of the face for which to return the landmarks. Defaults to 0.
            idxs (list, optional): List of indices of the landmarks to return. Defaults to None.

        Returns:
            list: List of face world landmarks.
        """
        if self.DETECTION_RESULT is not None:
            if idxs is None:
                return self.DETECTION_RESULT.face_landmarks[face_idx]
            else:
                return [
                    self.DETECTION_RESULT.face_landmarks[face_idx][idx] for idx in idxs
                ]
        else:
            return []

    @staticmethod
    def download_model() -> str:
        """
        Download the face_landmarker task model from the mediapipe repository.
            https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

        Returns:
            str: Path to the downloaded model.
        """
        root = os.path.dirname(os.path.realpath(__file__))
        # Unino to res folder
        root = os.path.join(root, "..", "res")
        filename = os.path.join(root, "face_landmarker.task")
        if os.path.exists(filename):
            print(f"O arquivo {filename} já existe, pulando o download.")
        else:
            base = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            urllib.request.urlretrieve(base, filename)

        return filename
