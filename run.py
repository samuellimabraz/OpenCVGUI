#!/usr/bin/env python3
import argparse
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
        choices=["tkinter", "streamlit", "pyside6", "qt"],
        default="tkinter",
        help="Choose the interface to run (tkinter, streamlit, pyside6/qt)",
    )

    args = parser.parse_args()

    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"

    if args.interface == "tkinter":
        print("Starting Tkinter interface...")
        tkinter_path = src_dir / "tkinter_app.py"
        subprocess.run([sys.executable, str(tkinter_path)])

    elif args.interface == "streamlit":
        print("Starting Streamlit interface...")
        streamlit_path = src_dir / "streamlit_app.py"
        subprocess.run(["streamlit", "run", str(streamlit_path)])

    elif args.interface in ["pyside6", "qt"]:
        print("Starting PySide6 (Qt6) interface...")
        pyside6_path = src_dir / "pyside6_app.py"
        subprocess.run([sys.executable, str(pyside6_path)])


if __name__ == "__main__":
    main()
