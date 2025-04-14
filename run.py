#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    """
    Main function to run the appropriate GUI application based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="OpenCV GUI Application Launcher")
    parser.add_argument(
        "-i",
        "--interface",
        choices=["tkinter", "streamlit"],
        default="tkinter",
        help="Choose the interface to run (tkinter or streamlit)",
    )

    args = parser.parse_args()

    # Get the absolute path to the src directory
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"

    if args.interface == "tkinter":
        print("Starting Tkinter interface...")
        # Run the tkinter application directly using Python
        tkinter_path = src_dir / "tkinter_app.py"
        subprocess.run([sys.executable, str(tkinter_path)])

    elif args.interface == "streamlit":
        print("Starting Streamlit interface...")
        # Run the streamlit application using the streamlit CLI
        streamlit_path = src_dir / "streamlit_app.py"
        subprocess.run(["streamlit", "run", str(streamlit_path)])


if __name__ == "__main__":
    main()
