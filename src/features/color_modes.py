"""
Color Mode Handler
Handles different color space conversions (BGR, Grayscale, HSV).
"""

import cv2

class ColorModeHandler:
    """Handles different color modes for the camera feed."""
    
    def __init__(self):
        """Initialize with default color mode."""
        self.current_mode = "COLOR"
        self.modes = {
            "COLOR": self._process_color,
            "GRAY": self._process_gray,
            "HSV": self._process_hsv
        }
    
    def set_mode(self, mode):
        """
        Set the current color mode.
        
        Args:
            mode (str): Color mode ("COLOR", "GRAY", or "HSV")
        """
        if mode in self.modes:
            self.current_mode = mode
            print(f"Color mode changed to: {mode}")
        else:
            print(f"Unknown color mode: {mode}")
    
    def apply(self, frame):
        """
        Apply the current color mode to a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Processed frame
        """
        if self.current_mode in self.modes:
            return self.modes[self.current_mode](frame)
        return frame
    
    def _process_color(self, frame):
        """
        Process frame in standard BGR color.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: BGR frame
        """
        return frame
    
    def _process_gray(self, frame):
        """
        Convert frame to grayscale.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Grayscale frame
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert back to 3-channel for consistent display
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return frame
    
    def _process_hsv(self, frame):
        """
        Convert frame to HSV color space.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: HSV frame
        """
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return frame
    
    def get_channel_count(self):
        """
        Get the number of channels for the current mode.
        
        Returns:
            int: Number of channels (1 for grayscale, 3 for color)
        """
        return 1 if self.current_mode == "GRAY" else 3