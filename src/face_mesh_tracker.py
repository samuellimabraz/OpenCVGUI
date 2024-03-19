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

        self.fps_avg_frame_count = 30

        self.COUNTER, self.FPS = 0, 0
        self.START_TIME = time.time()
        self.DETECTION_RESULT = None

    def save_result(
        self,
        result: vision.FaceLandmarkerResult,
        unused_output_image,
        timestamp_ms: int,
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
        if self.COUNTER % self.fps_avg_frame_count == 0:
            self.FPS = self.fps_avg_frame_count / (time.time() - self.START_TIME)
            self.START_TIME = time.time()

        self.DETECTION_RESULT = result
        self.COUNTER += 1

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
        # Show the FPS
        fps_text = "FPS = {:.1f}".format(self.FPS)
        cv2.putText(
            image,
            fps_text,
            (24, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            font_size,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

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
    def download_model():
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


# class EyeTrackerController:
#     def __init__(self, face_tracker: FaceMeshTracker, eye_idx: int = 0):
#         """
#         Initialize an EyeTrackerController instance.

#         Args:
#             face_tracker (FaceTracker): FaceTracker instance.
#             eye_idx (int, optional): Index of the eye for which to track the gaze. Defaults to 0.
#                 0 - Left eye
#                 1 - Right eye
#         """
#         self.tracker = face_tracker
#         self.eye_indices = (
#             FaceMeshTracker.LEFT_EYE + FaceMeshTracker.LEFT_CENTER_EYE
#             if eye_idx == 0
#             else FaceMeshTracker.RIGHT_EYE + FaceMeshTracker.RIGHT_CENTER_EYE
#         )
#         self.wScr, self.hScr = autopy.screen.size()

#     def get_eye_region(self, image, landmarks):
#         """
#         Returns the eye region of the face.

#         Args:
#             image (numpy.ndarray): Image on which to draw the landmarks.
#             landmarks (list): List of face landmarks.
#         Returns:
#             numpy.ndarray: Eye region of the face.
#         """
#         threshold = 0.1
#         landmarks = np.array(
#             [
#                 (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
#                 for landmark in landmarks
#             ]
#         )
#         x, y, w, h = cv2.boundingRect(landmarks)

#         # Plus a threshhold to size of bounding box
#         x -= int(w * threshold)
#         y -= int(h * threshold)
#         w += int(w * threshold * 2)
#         h += int(h * threshold * 2)

#         return image[y : y + h, x : x + w], (x, y, w, h)

#     def get_eye_region_center(self, eye_region):
#         rows, cols, _ = eye_region.shape
#         gray_roi = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
#         gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

#         gray_roi = cv2.equalizeHist(gray_roi)

#         _, threshold = cv2.threshold(gray_roi, 45, 255, cv2.THRESH_BINARY_INV)

#         contours, _ = cv2.findContours(
#             threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
#         )
#         contour = max(contours, key=cv2.contourArea)

#         # Calculate the moments of the largest contour.
#         moments = cv2.moments(contour)

#         # Calculate the centroid of the iris.
#         cX = int(moments["m10"] / moments["m00"])
#         cY = int(moments["m01"] / moments["m00"])

#         # Draw a circle at the center.
#         cv2.circle(eye_region, (cX, cY), 2, (0, 255, 0), -1)

#         # Draw a horizontal line across the center.
#         cv2.line(eye_region, (0, cY), (eye_region.shape[1], cY), (0, 255, 0), 1)

#         # Draw a vertical line down the center.
#         cv2.line(eye_region, (cX, 0), (cX, eye_region.shape[0]), (0, 255, 0), 1)

#         return cX, cY

#     def process(self, image, face_idx=0):
#         """
#         Processes the image to detect the eyes.

#         Args:
#             image (numpy.ndarray): Image on which to draw the landmarks.
#             face_idx (int, optional): Index of the face for which to return the landmarks. Defaults to 0.

#         Returns:
#             numpy.ndarray: Processed image.
#         """
#         cv2.flip(image, 1, image)
#         image = self.tracker.detect(image, draw=False)
#         eye_region = None
#         face_landmarks = self.tracker.get_face_landmarks(face_idx, self.eye_indices)
#         if len(face_landmarks) != 0:
#             eye_region, (x, y, w, h) = self.get_eye_region(image, face_landmarks)
#             # Draw bounding box around the eye region
#             eye_region = cv2.resize(
#                 eye_region, (int(image.shape[1]), int(image.shape[0]))
#             )
#             # Get the center of the eye region
#             eye_center = self.get_eye_region_center(eye_region)
#             if eye_center is not None:
#                 print(eye_center)
#                 cX, cY = eye_center

#                 cX = np.interp(cX, (0, eye_region.shape[1]), (0, self.wScr))
#                 cY = np.interp(cY, (0, eye_region.shape[0]), (0, self.hScr))

#                 # Move the mouse cursor to the normalized position.
#                 autopy.mouse.move(cX, cY)

#             cv2.imshow("eye_region", eye_region)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             self.tracker.draw_landmark_circles(image, self.eye_indices)
#         return image


def main():
    tracker = FaceMeshTracker(
        num_faces=1,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    # controller = EyeTrackerController(tracker)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                "ERROR: Unable to read from webcam. Please verify your webcam settings."
            )

        try:
            # image = controller.process(image)
            image = tracker.detect(image, draw=True)
        except Exception as e:
            print(e)
            break

        cv2.imshow("hand_landmarker", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
