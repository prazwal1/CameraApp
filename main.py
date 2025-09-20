
"""
Main entry point for the Advanced Camera Application.
"""

from src.camera_app import CameraApp

def main():
    """Initialize and run the camera application."""
    app = CameraApp()
    app.run()

if __name__ == "__main__":
    main()