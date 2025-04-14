# OpenCV GUI with Gradio and FastRTC

A real-time video processing application that demonstrates various OpenCV functions using a modern web interface.

## Features

- Real-time video streaming with FastRTC
- Multiple image processing filters and effects:
  - Color filtering with HSV controls
  - Edge detection with Canny algorithm
  - Gaussian blur with adjustable kernel size
  - Image rotation
  - Image resizing
  - Contour detection
  - Hand tracking using MediaPipe
  - Face mesh tracking using MediaPipe
- Order-dependent processing pipeline
- FPS display
- User-friendly web interface with Gradio

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

Run the application with:

```bash
python src/run.py
```

By default, this will launch the Stramlit interface. You can specify which interface to use:

```bash
# Launch with Streamlit interface 
python src/run.py -i stramlit

# Launch with Tkinter interface (default)
python src/run.py -i tkinter
```

## How It Works

The application captures video from your webcam and applies various OpenCV transformations in real-time based on your selections. The processing is sequential, meaning the order of operations matters - each operation is applied to the result of the previous one.

## Interface

The interface is divided into two main sections:
- Right: Video display showing the processed feed from your webcam
- Left: Control panel with various image processing options in collapsible accordions

Each processing option can be enabled/disabled and configured with sliders and other controls.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


