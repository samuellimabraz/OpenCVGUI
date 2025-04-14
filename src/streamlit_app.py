import av
import cv2
import numpy as np
import streamlit as st
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from opencv_utils import OpenCVUtils
from twilio.rest import Client

st.set_page_config(page_title="OpenCV Explorer", page_icon="🎨", layout="wide")


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
                "Twilio credentials not found. Using free STUN server only, which may not work reliably on Streamlit Cloud."
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
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
# ---------------------------

st.markdown("# 🎨 OpenCV Explorer")
st.markdown("Explore filters and transformations in real-time using your webcam.")

# Sidebar Controls
FUNCTIONS = [
    "Color Filter",
    "Canny",
    "Blur",
    "Rotation",
    "Resize",
    "Contour",
    "Histogram Equalization",
    "Adaptive Threshold",
    "Morphology",
    "Sharpen",
    "Hough Lines",
    "Optical Flow",
    "Pencil Sketch",
    "Color Quantization",
    "Hand Tracker",
    "Face Tracker",
]
selected_functions = st.sidebar.multiselect(
    "Select and order functions:", FUNCTIONS, default=[]
)
# Parameters
with st.sidebar.expander("Color Filter"):
    lh = st.slider("Lower Hue", 0, 180, 0)
    uh = st.slider("Upper Hue", 0, 180, 180)
    ls = st.slider("Lower Sat", 0, 255, 0)
    us = st.slider("Upper Sat", 0, 255, 255)
    lv = st.slider("Lower Val", 0, 255, 0)
    uv = st.slider("Upper Val", 0, 255, 255)
with st.sidebar.expander("Canny Edge"):
    lc = st.slider("Lower Canny", 0, 255, 100)
    uc = st.slider("Upper Canny", 0, 255, 200)
with st.sidebar.expander("Blur"):
    bk = st.slider("Kernel Size (odd)", 1, 15, 5, step=2)
with st.sidebar.expander("Rotation"):
    ang = st.slider("Angle", 0, 360, 0)
with st.sidebar.expander("Resize"):
    w = st.slider("Width", 100, 1920, 640)
    h = st.slider("Height", 100, 1080, 480)
with st.sidebar.expander("Morphology"):
    morph_op = st.selectbox("Operation", ["erode", "dilate", "open", "close"])
    morph_ks = st.slider("Kernel Size", 1, 31, 5, step=2)

prev_gray = None


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global prev_gray
    img = frame.to_ndarray(format="bgr24")
    curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    prev_gray = curr_gray
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="opencv-explorer",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
