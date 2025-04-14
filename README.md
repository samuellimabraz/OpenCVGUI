# OpenCV Explorer

A real-time web application for exploring OpenCV filters and transformations using your webcam through Streamlit.

## Features

- Real-time video processing using WebRTC
- Multiple image filters and transformations
- Hand tracking and face mesh detection with MediaPipe
- Support for both local development and cloud deployment

## Filters and Transformations

- Color filtering with HSV controls
- Edge detection with Canny algorithm
- Gaussian blur with adjustable kernel size
- Image rotation
- Image resizing
- Contour detection
- Histogram equalization
- Adaptive thresholding
- Morphological operations
- Sharpening
- Hough line detection
- Optical flow visualization
- Pencil sketch effect
- Color quantization
- Hand tracking
- Face mesh tracking

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/OpenCVGUI.git
   cd OpenCVGUI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application locally with:

```bash
python run.py -i streamlit
```

Or directly with Streamlit:

```bash
streamlit run src/streamlit_app.py
```

## Deployment to Streamlit Cloud

When deploying to Streamlit Cloud, you need to configure a TURN server to ensure WebRTC connections work properly. This application uses Twilio's TURN server service. Follow these steps:

1. Create a free Twilio account at [https://www.twilio.com/try-twilio](https://www.twilio.com/try-twilio)
2. Once you have your account, go to your Twilio Console dashboard
3. Find your Account SID and Auth Token
4. Set these values as secrets in your Streamlit Cloud deployment:
   - Go to your app settings in Streamlit Cloud
   - Add two secrets:
     - `TWILIO_ACCOUNT_SID` with your Account SID value
     - `TWILIO_AUTH_TOKEN` with your Auth Token value

Twilio offers a free trial with a certain amount of credit, which should be sufficient for testing and moderate usage.

## How It Works

The application captures video from your webcam and applies various OpenCV transformations in real-time based on your selections. The processing is sequential, meaning the order of operations matters - each operation is applied to the result of the previous one.

The WebRTC component ensures low-latency video streaming, making the transformations appear in real-time.

## Troubleshooting

- If you encounter connection issues on Streamlit Cloud, check that your Twilio credentials are correctly set up.
- If the webcam doesn't start, ensure your browser has permission to access the camera.
- Some browsers or networks may have restrictive policies that block WebRTC connections. Try a different browser or network if you encounter issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


