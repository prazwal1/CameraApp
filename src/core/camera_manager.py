"""
Camera Manager
Handles camera initialization and frame capture.
"""

import cv2

class CameraManager:
    """Manages camera operations and frame capture."""
    
    def __init__(self, camera_index=0):
        """
        Initialize camera.
        
        Args:
            camera_index (int): Camera device index (default: 0)
        """
        self.camera_index = camera_index
        self.cap = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {self.camera_index}")
        
        # Set some camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def get_frame(self):
        """
        Capture and return a frame from the camera.
        
        Returns:
            numpy.ndarray or None: Camera frame or None if capture failed
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def is_opened(self):
        """
        Check if camera is opened and ready.
        
        Returns:
            bool: True if camera is ready, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_properties(self):
        """
        Get current camera properties.
        
        Returns:
            dict: Dictionary of camera properties
        """
        if not self.is_opened():
            return {}
        
        properties = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION)
        }
        return properties