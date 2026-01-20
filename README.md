# OpenCV Explorer

A versatile application for exploring OpenCV filters and transformations using your webcam, available with Streamlit web interface, Tkinter desktop interface, and a modern PySide6 (Qt6) desktop interface.

## Features

- Real-time video processing with WebRTC (Streamlit), native camera capture (Tkinter), and Qt6 (PySide6)
- Multiple image filters and transformations
- Hand tracking and face mesh detection with MediaPipe
- Support for both local development and cloud deployment
- ArUco marker detection with multiple dictionary options
- Easy-to-use UI with adjustable parameters for each filter
- **Modern Qt6 interface** with collapsible sections, smooth animations, and a beautiful dark theme

## Filters and Transformations

- Color filtering with HSV controls
- Edge detection with Canny algorithm
- Gaussian blur with adjustable kernel size
- Image rotation
- Image resizing
- Contour detection
- Histogram equalization
- Adaptive thresholding
- Morphological operations (erosion, dilation, opening, closing)
- Sharpening
- Hough line detection
- Hough circle detection
- Optical flow visualization
- Pencil sketch effect
- Color quantization
- Stylization effect
- Cartoonify effect
- Hand tracking with MediaPipe
- Face mesh tracking with MediaPipe
- ArUco marker detection with multiple dictionary options

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

### PySide6 (Qt6) Desktop Interface (Recommended)

Run the modern Qt6 interface:

```bash
python run.py -i pyside6
```

Or using the alias:

```bash
python run.py -i qt
```

Or directly:

```bash
python src/pyside6_app.py
```

The PySide6 interface features:
- Modern dark theme with teal accents
- Collapsible filter sections for better organization
- Smooth slider controls with real-time value display
- Responsive layout with adjustable splitter
- High-performance video rendering

### Streamlit Web Interface

Run the application with Streamlit:

```bash
python run.py -i streamlit
```

Or directly:

```bash
streamlit run src/streamlit_app.py
```

### Tkinter Desktop Interface

Run the application with Tkinter:

```bash
python run.py -i tkinter
```

Or directly:

```bash
python src/tkinter_app.py
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

### Streamlit Interface
The WebRTC component ensures low-latency video streaming in the browser, making the transformations appear in real-time.

### Tkinter Interface
The native desktop interface provides direct access to your webcam with minimal latency, perfect for testing OpenCV functions locally.

## ArUco Marker Detection

The application includes support for detecting ArUco markers, which are useful for augmented reality, camera calibration, and pose estimation. You can select from multiple ArUco dictionaries:

- 4x4 markers (50, 100, 250, 1000 markers)
- 5x5 markers (50, 100, 250, 1000 markers)
- 6x6 markers (50, 100, 250, 1000 markers)
- 7x7 markers (50, 100, 250, 1000 markers)
- Original ArUco markers

## Troubleshooting

- If you encounter connection issues on Streamlit Cloud, check that your Twilio credentials are correctly set up.
- If the webcam doesn't start, ensure your browser has permission to access the camera.
- Some browsers or networks may have restrictive policies that block WebRTC connections. Try a different browser or network if you encounter issues.
- For Tkinter interface, press 'q' to exit the application.
- For PySide6 interface, close the window normally or press Ctrl+Q to exit.
- If PySide6 fails to start, ensure you have Qt6 libraries installed: `pip install PySide6`

## License

This project is licensed under the MIT License - see the LICENSE file for details.


