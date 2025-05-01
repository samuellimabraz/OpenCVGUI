import av
import cv2
import numpy as np
import streamlit as st
import os
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoHTMLAttributes
from opencv_utils import OpenCVUtils
from twilio.rest import Client

# Custom theme settings
st.set_page_config(
    page_title="OpenCV Explorer",
    page_icon="⚫",  # Changed icon for minimalism
    layout="wide",
    initial_sidebar_state="expanded",
)


# Create a custom theme
def create_custom_theme():
    # Create a .streamlit directory if it doesn't exist
    os.makedirs(".streamlit", exist_ok=True)
    # Create a config.toml file with custom theme settings
    with open(".streamlit/config.toml", "w") as f:
        f.write(
            """
[theme]
base = "dark" # Use Streamlit's dark theme as a base
primaryColor = "#CCCCCC"  # Light Grey accent
backgroundColor = "#0E1117" # Default Streamlit dark bg
secondaryBackgroundColor = "#262730" # Slightly lighter dark grey
textColor = "#FAFAFA" # Light text
font = "sans serif"
        """
        )


# Apply custom theme
create_custom_theme()


def get_ice_servers():
    """
    Get ICE servers configuration.
    For Streamlit Cloud deployment, a TURN server is required in addition to STUN.
    This function will try to use Twilio's TURN server service if credentials are available,
    otherwise it falls back to a free STUN server from Google.
    """
    try:
        # Try to get Twilio credentials from environment variables
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

        if account_sid and auth_token:
            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return token.ice_servers
        else:
            st.warning(
                "Twilio credentials not found. Using free STUN server only, which may not work reliably."  # Removed Streamlit Cloud mention for generality
            )
    except Exception as e:
        st.error(f"Error setting up Twilio TURN servers: {e}")

    # Fallback to Google's free STUN server
    return [{"urls": ["stun:stun.l.google.com:19302"]}]


@st.cache_resource
def get_app():
    return OpenCVUtils()


app = get_app()

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            /* Body background - Already set by theme config */
            /* .stApp {
                background-color: #0E1117; 
            } */
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px; /* Slightly reduced gap */
                border-bottom: 1px solid #333333; /* Darker border */
            }
            .stTabs [data-baseweb="tab"] {
                background-color: transparent; /* Make tabs transparent */
                border-radius: 0; /* Remove border radius */
                padding: 10px 15px;
                color: #AAAAAA; /* Lighter Grey text */
                border-bottom: 2px solid transparent; /* Prepare for selected indicator */
                transition: all 0.3s ease;
            }
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #262730; /* Dark grey hover */
                color: #FAFAFA; /* White text on hover */
            }
            .stTabs [aria-selected="true"] {
                background-color: transparent !important;
                color: #FAFAFA !important; /* White text for selected */
                border-bottom: 2px solid #CCCCCC !important; /* Light grey underline for selected */
                font-weight: 600; /* Make selected tab bold */
            }
            /* Sidebar styling - Mostly handled by theme config */
            /* .css-1d391kg { 
                 background-color: #262730 !important; 
            } */
             /* Ensure sidebar text is readable - Mostly handled by theme config */
            /* .css-1d391kg .stMarkdown, .css-1d391kg .stCheckbox, .css-1d391kg .stExpander, .css-1d391kg .stText, .css-1d391kg .stButton > button {
                color: #FAFAFA !important;
            } */
            /* Button styling */
            .stButton>button {
                background-color: #262730 !important; /* Dark grey background */
                color: #FAFAFA !important; /* Light text */
                border: 1px solid #444444 !important; /* Slightly lighter border */
                transition: all 0.3s ease !important;
                box-shadow: none !important; /* Remove default shadow */
            }
            .stButton>button:hover {
                background-color: #3a3c44 !important; /* Lighter grey on hover */
                border-color: #666666 !important;
                transform: none !important; /* Remove hover transform */
                box-shadow: none !important;
            }
            .stButton>button:active {
                 background-color: #444444 !important; /* Slightly darker grey on click */
            }
             /* Expander header styling */
            .stExpander > div:first-child {
                background-color: #262730; /* Dark grey background for expander header */
                border-radius: 4px;
            }
            .stExpander header { /* Target expander header specifically */
                color: #FAFAFA !important; /* Light text for expander header */
                font-weight: 600;
            }
            /* General adjustments for minimalist feel */
            h1, h2, h3, h4, h5, h6 {
                color: #FAFAFA; /* Ensure headers are light */
            }
            .stMarkdown p {
                color: #CCCCCC; /* Slightly darker light color for paragraph text */
            }
            /* Color Preview Text */
            .color-preview-text {
                color: #1E1E1E !important; /* Ensure text is visible on light/dark previews */
                text-shadow: 0 0 2px #FFFFFF; /* Add a subtle white shadow for contrast */
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
# ---------------------------

# Top header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image(
        "https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_black-2.png",
        width=100,
    )
with col2:
    st.title("OpenCV Explorer")
    st.markdown(
        """
    <p style='font-size: 18px; margin-top: -10px;'>
    Explore computer vision filters and transformations in real-time using your webcam.
    </p>
    """,
        unsafe_allow_html=True,
    )

main_tabs = st.tabs(["📹 Camera Feed", "ℹ️ About", "📋 Documentation"])
with main_tabs[0]:  # Camera Feed Tab
    # Create columns for camera and controls
    video_col, control_col = st.columns([3, 1])
    with control_col:
        st.markdown("## 🎛️ Controls")

        # List all available filters
        all_filters = [
            "Resize",
            "Rotation",
            "Blur",
            "Sharpen",
            "Canny",
            "Contour",
            "Hough Lines",
            "Color Filter",
            "Histogram Equalization",
            "Color Quantization",
            "Pencil Sketch",
            "Morphology",
            "Adaptive Threshold",
            "Optical Flow",
            "Hand Tracker",
            "Face Tracker",
            "ArUco Marker Detector",
        ]

        # Use multiselect to both select and order filters
        selected_functions = st.multiselect(
            "Select and order filters to apply:",
            options=all_filters,
            default=[],
            help="Filters will be applied in the order they appear here. Drag to reorder.",
        )

        # Show the currently applied filters with their order
        if selected_functions:
            st.markdown("### 📌 Applied Filters")
            for i, fn in enumerate(selected_functions):
                st.markdown(f"**{i+1}.** {fn}")
        else:
            st.info("Select filters to apply to the camera feed")

        # Filter parameters - using expanders for cleaner UI
        if "Resize" in selected_functions:
            with st.expander("📐 Resize Parameters", expanded=True):
                w = st.slider("Width", 320, 1920, 1280)
                h = st.slider("Height", 240, 1080, 720)
        else:
            # Default values if not displayed
            w, h = 1280, 720

        if "Rotation" in selected_functions:
            with st.expander("🔄 Rotation Parameters", expanded=True):
                ang = st.slider("Angle", 0, 360, 0)
        else:
            ang = 0

        if "Blur" in selected_functions:
            with st.expander("🌫️ Blur Parameters", expanded=True):
                bk = st.slider("Kernel Size (odd)", 1, 15, 5, step=2)
        else:
            bk = 5

        if "Color Filter" in selected_functions:
            with st.expander("🎨 Color Filter Parameters", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Lower Bounds**")
                    lh = st.slider("Hue (L)", 0, 180, 0)
                    ls = st.slider("Sat (L)", 0, 255, 0)
                    lv = st.slider("Val (L)", 0, 255, 0)
                with col2:
                    st.markdown("**Upper Bounds**")
                    uh = st.slider("Hue (U)", 0, 180, 180)
                    us = st.slider("Sat (U)", 0, 255, 255)
                    uv = st.slider("Val (U)", 0, 255, 255)

        else:
            lh, ls, lv, uh, us, uv = 0, 0, 0, 180, 255, 255

        if "Canny" in selected_functions:
            with st.expander("📊 Canny Edge Parameters", expanded=True):
                lc = st.slider("Lower Threshold", 0, 255, 100)
                uc = st.slider("Upper Threshold", 0, 255, 200)
        else:
            lc, uc = 100, 200

        if "Morphology" in selected_functions:
            with st.expander("🧩 Morphology Parameters", expanded=True):
                morph_op = st.selectbox(
                    "Operation", ["erode", "dilate", "open", "close"]
                )
                morph_ks = st.slider("Kernel Size", 1, 31, 5, step=2)
        else:
            morph_op, morph_ks = "erode", 5

        if "ArUco Marker Detector" in selected_functions:
            with st.expander("🔍 ArUco Marker Parameters", expanded=True):
                aruco_dict = st.selectbox(
                    "ArUco Dictionary",
                    options=[
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
                    ],
                    index=10,  # Default to DICT_6X6_250
                    help="Select the ArUco marker dictionary. Different dictionaries support different marker patterns and IDs.",
                )
        else:
            aruco_dict = "DICT_6X6_250"

    with video_col:
        st.markdown("## 📹 Live Camera Feed")
        # WebRTC settings for real-time video
        prev_gray = None

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            global prev_gray
            img = frame.to_ndarray(format="bgr24")
            curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply filters in the order they were selected
            for fn in selected_functions:
                if fn == "Color Filter":
                    img = app.apply_color_filter(img, (lh, ls, lv), (uh, us, uv))
                elif fn == "Canny":
                    img = app.apply_edge_detection(img, lc, uc)
                elif fn == "Blur":
                    img = app.blur_image(img, bk)
                elif fn == "Rotation":
                    img = app.rotate_image(img, ang)
                elif fn == "Resize":
                    img = app.resize_image(img, w, h)
                elif fn == "Contour":
                    img = app.apply_contour_detection(img)
                elif fn == "Histogram Equalization":
                    img = app.equalize_histogram(img)
                elif fn == "Adaptive Threshold":
                    img = app.adaptive_threshold(img)
                elif fn == "Morphology":
                    img = app.morphology(img, morph_op, morph_ks)
                elif fn == "Sharpen":
                    img = app.sharpen(img)
                elif fn == "Hough Lines":
                    img = app.hough_lines(img)
                elif fn == "Optical Flow" and prev_gray is not None:
                    img = app.optical_flow(prev_gray, curr_gray, img)
                elif fn == "Pencil Sketch":
                    img = app.pencil_sketch(img)
                elif fn == "Color Quantization":
                    img = app.color_quantization(img)
                elif fn == "Hand Tracker":
                    img = app.detect_hands(img)
                elif fn == "Face Tracker":
                    img = app.detect_faces(img)
                elif fn == "ArUco Marker Detector":
                    img = app.detect_aruco_markers(img, dict_type=aruco_dict)

            prev_gray = curr_gray
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="opencv-explorer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_html_attrs=VideoHTMLAttributes(
                autoPlay=True,
                controls=False,
                style={
                    "width": f"{w}px",
                    "height": f"{h}px",
                    "border-radius": "8px",
                    "margin": "0 auto",
                    "display": "block",
                    "border": "2px solid #AAAAAA",
                },
            ),
        )

        # Performance metrics
        with st.expander("📊 Performance Metrics", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Resolution", f"{w}x{h} px")
            col2.metric("Filters Applied", len(selected_functions))
            col3.metric("Frame Processing", f"{time.time():.2f} ms", delta=None)

with main_tabs[1]:  # About Tab
    st.markdown(
        """
    ## About OpenCV Explorer
    
    OpenCV Explorer is an interactive web application that allows you to experiment with various computer vision techniques in real-time using your webcam. This application is built with:
    
    - **OpenCV**: Open Source Computer Vision Library
    - **Streamlit**: An open-source app framework for Machine Learning and Data Science
    - **WebRTC**: Web Real-Time Communication for live video streaming
    
    ### Features
    
    - Apply multiple filters and transformations to your webcam feed
    - Adjust parameters in real-time
    - Experiment with advanced computer vision techniques
    - Learn about image processing concepts
    
    ### How to Use
    
    1. Select one or more filters from the categories in the control panel
    2. Adjust the parameters for each selected filter
    3. See the results in real-time through your webcam
    4. Reorder filters to create different effects
    
    ### Privacy Note
    
    All processing is done in your browser. No video data is sent to any server except for the WebRTC connection.
    """
    )

with main_tabs[2]:  # Documentation Tab
    st.markdown(
        """
    ## Documentation

    This section provides details about the available computer vision filters and transformations.
    You can select multiple filters, and they will be applied sequentially in the order chosen within each category.
    Adjust the parameters in the control panel to see the effects in real-time.

    ### Available Filters
    """
    )

    # Create documentation for each filter category
    for filter_name in all_filters:
        st.markdown(f"#### {filter_name}")

        # Add detailed description and links for each filter
        if filter_name == "Resize":
            st.markdown(
                """
            Changes the dimensions (width and height) of the video frame. Useful for adjusting the output size or preparing the frame for other operations that require a specific input size.

            **Parameters:**
            - **Width**: Target width in pixels.
            - **Height**: Target height in pixels.

            **Usage**: Scaling for performance, UI fitting, preprocessing for models.

            **Docs**: [OpenCV Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html) (See `cv2.resize`)
            """
            )
        elif filter_name == "Rotation":
            st.markdown(
                """
            Rotates the video frame around its center by a specified angle.

            **Parameters:**
            - **Angle**: Rotation angle in degrees (0-360).

            **Usage**: Image orientation correction, creative effects.

            **Docs**: [OpenCV Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html) (See `cv2.getRotationMatrix2D` and `cv2.warpAffine`)
            """
            )
        elif filter_name == "Blur":
            st.markdown(
                """
            Applies Gaussian blur to smooth the image, reducing noise and detail. The kernel size determines the extent of blurring.

            **Parameters:**
            - **Kernel Size**: Size of the blurring matrix (must be an odd number). Higher values create more blur.

            **Usage**: Noise reduction, detail smoothing, pre-processing for edge detection or other algorithms.

            **Docs**: [OpenCV Smoothing Images](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) (See `cv2.GaussianBlur`)
            """
            )
        elif filter_name == "Sharpen":
            st.markdown(
                """
            Enhances the edges and details in the image using a sharpening kernel. This is achieved by subtracting a blurred version of the image from the original.

            **Parameters:** None (uses a fixed kernel).

            **Usage**: Enhancing image clarity, highlighting details.

            **Docs**: [OpenCV Image Filtering Concepts](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) (Concept explanation, the implementation uses a custom kernel)
            """
            )
        elif filter_name == "Canny":
            st.markdown(
                """
            Detects edges in the image using the Canny edge detection algorithm, a multi-stage process to find sharp changes in intensity.

            **Parameters:**
            - **Lower Threshold**: Minimum intensity gradient to be considered a potential edge.
            - **Upper Threshold**: Maximum intensity gradient. Edges above this are definite edges. Pixels between the thresholds are included if connected to definite edges.

            **Usage**: Edge detection, feature extraction, object boundary identification.

            **Docs**: [OpenCV Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
            """
            )
        elif filter_name == "Contour":
            st.markdown(
                """
            Finds and draws contours (continuous curves joining points along a boundary with the same intensity) in the image. Usually applied after thresholding or edge detection.

            **Parameters:** None (finds contours on the processed image and draws them).

            **Usage**: Object detection, shape analysis, feature extraction.

            **Docs**: [OpenCV Contours](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html) (See `cv2.findContours`, `cv2.drawContours`)
            """
            )
        elif filter_name == "Hough Lines":
            st.markdown(
                """
            Detects straight lines in the image using the Hough Line Transform (Probabilistic variant). Works best on edge-detected images.

            **Parameters:** None (uses preset parameters for `cv2.HoughLinesP`).

            **Usage**: Line detection in images, structure identification.

            **Docs**: [OpenCV Hough Line Transform](https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html) (See `cv2.HoughLinesP`)
            """
            )
        elif filter_name == "Color Filter":
            st.markdown(
                """
            Isolates specific colors by converting the image to HSV (Hue, Saturation, Value) color space and applying a threshold based on the selected ranges.

            **Parameters:**
            - **Lower Bounds (Hue, Sat, Val)**: Minimum HSV values for the color range.
            - **Upper Bounds (Hue, Sat, Val)**: Maximum HSV values for the color range.

            **Usage**: Object detection based on color, color segmentation, special effects.

            **Docs**: [OpenCV Changing Colorspaces](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html) (See `cv2.cvtColor` and `cv2.inRange`)
            """
            )
        elif filter_name == "Histogram Equalization":
            st.markdown(
                """
            Improves contrast in grayscale images by redistributing pixel intensities more evenly across the histogram. Applied to the Value channel if the input is color.

            **Parameters:** None.

            **Usage**: Enhancing contrast in low-contrast images, improving visibility of details.

            **Docs**: [OpenCV Histogram Equalization](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) (See `cv2.equalizeHist`)
            """
            )
        elif filter_name == "Color Quantization":
            st.markdown(
                """
            Reduces the number of distinct colors in an image using K-Means clustering in the color space. Groups similar colors together.

            **Parameters:** None (uses a fixed number of clusters, K=8).

            **Usage**: Image compression, posterization effect, simplifying color palettes.

            **Docs**: [OpenCV K-Means Clustering](https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html) (Underlying algorithm)
            """
            )
        elif filter_name == "Pencil Sketch":
            st.markdown(
                """
            Creates a pencil sketch effect by converting the image to grayscale, inverting it, blurring the inverted image, and blending it with the original grayscale image using color dodge.

            **Parameters:** None.

            **Usage**: Artistic image transformation, creating sketch-like visuals.

            **Docs**: Involves multiple OpenCV steps (Grayscale, Blur, Blending). See [Color Dodge Blending](https://en.wikipedia.org/wiki/Blend_modes#Dodge_and_burn).
            """
            )
        elif filter_name == "Morphology":
            st.markdown(
                """
            Applies morphological operations (Erode, Dilate, Open, Close) to modify the shape of features in the image, typically on binary images.

            **Parameters:**
            - **Operation**: Type of morphological operation (`erode`, `dilate`, `open`, `close`).
            - **Kernel Size**: Size of the structuring element used (odd number).

            **Usage**: Noise removal, joining broken parts, thinning/thickening features.

            **Docs**: [OpenCV Morphological Transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) (See `cv2.erode`, `cv2.dilate`, `cv2.morphologyEx`)
            """
            )
        elif filter_name == "Adaptive Threshold":
            st.markdown(
                """
            Applies adaptive thresholding, where the threshold value is calculated locally for different regions of the image. Useful for images with varying illumination.

            **Parameters:** None (uses `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`).

            **Usage**: Image segmentation in non-uniform lighting conditions.

            **Docs**: [OpenCV Image Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) (See `cv2.adaptiveThreshold`)
            """
            )
        elif filter_name == "Optical Flow":
            st.markdown(
                """
                Calculates and visualizes the apparent motion of objects between consecutive frames using the Farneback algorithm. Shows motion vectors as lines on the image.

                **Parameters:** None (Requires previous frame data internally).

                **Usage**: Motion tracking, video stabilization analysis, action recognition.

                **Docs**: [OpenCV Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) (See `cv2.calcOpticalFlowFarneback`)
                """
            )
        elif filter_name == "Hand Tracker":
            st.markdown(
                """
            Detects and tracks hand positions and landmarks (joints) in real-time using the MediaPipe Hands solution. Draws landmarks and connections on the detected hands.

            **Parameters:** None (uses pre-trained MediaPipe models).

            **Usage**: Gesture recognition, sign language interpretation, virtual object interaction, hand pose estimation.

            **Docs**: [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
            """
            )
        elif filter_name == "Face Tracker":
            st.markdown(
                """
            Detects faces in the video feed using the MediaPipe Face Detection solution and draws bounding boxes around them.

            **Parameters:** None (uses pre-trained MediaPipe models).

            **Usage**: Face detection, counting people, basic facial analysis applications, input for face recognition or landmark detection.

            **Docs**: [MediaPipe Face Detector](https://developers.google.com/mediapipe/solutions/vision/face_detector)
            """
            )
        elif filter_name == "ArUco Marker Detector":
            st.markdown(
                """
            Detects ArUco markers in the video feed. ArUco markers are square fiducial markers that can be used for camera pose estimation, calibration, and object tracking.

            **Parameters:**
            - **ArUco Dictionary**: Select the dictionary type for the markers you want to detect. Different dictionaries support different marker patterns and ID ranges.

            **Usage**: 
            - Augmented reality
            - Camera calibration
            - Object tracking 
            - Robotics navigation
            - Positional reference

            **How it works**:
            1. Converts the image to grayscale
            2. Detects markers using the selected dictionary
            3. Draws detected markers with their IDs
            
            **Docs**: [OpenCV ArUco Marker Detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
            """
            )
        else:
            # Fallback for any filters missed
            st.markdown(
                f"Detailed documentation for the **{filter_name}** filter is pending."
            )

        st.divider()  # Add a separator between filter descriptions

    st.markdown(
        """
    ### General Technical Details

    - **OpenCV**: The core library used for most image processing functions. [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
    - **MediaPipe**: Used for the advanced Hand and Face Tracking features. [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/)
    - **Streamlit**: The framework used to build this web application interface. [Streamlit Documentation](https://docs.streamlit.io/)
    - **WebRTC**: Enables real-time video streaming from your webcam to the browser for processing. (Handled by `streamlit-webrtc`)
    """
    )

st.markdown(
    """
<div style="position: fixed; bottom: 0; width: 100%; background-color: #0E1117;
            padding: 8px; text-align: center; border-top: 1px solid #262730;">
    <p style="margin: 0; font-size: 13px; color: #AAAAAA;">
    OpenCV Explorer | Built with Streamlit 
    </p>
</div>
""",
    unsafe_allow_html=True,
)
