"""
Edge Detection Handler
Handles Canny edge detection operations.
"""

import cv2

class EdgeDetectionHandler:
    """Handles Canny edge detection."""
    
    def __init__(self):
        """Initialize edge detection state."""
        self.active = False
    
    def set_active(self, active):
        """
        Set the active state of edge detection.
        
        Args:
            active (bool): Whether edge detection is active
        """
        self.active = active
    
    def is_active(self):
        """
        Check if edge detection is active.
        
        Returns:
            bool: True if active, False otherwise
        """
        return self.active
    
    def create_trackbars(self, window_name, trackbar_manager):
        """
        Create trackbars for Canny edge detection parameters.
        
        Args:
            window_name (str): Name of the window
            trackbar_manager: Trackbar manager instance
        """
        trackbar_manager.create_trackbar('Low Threshold', window_name, 50, 500)
        trackbar_manager.create_trackbar('High Threshold', window_name, 150, 500)
        trackbar_manager.create_trackbar('Aperture Size', window_name, 3, 7)
    
    def apply(self, frame, window_name):
        """
        Apply Canny edge detection if active.
        
        Args:
            frame (numpy.ndarray): Input frame
            window_name (str): Name of the window for trackbar access
            
        Returns:
            numpy.ndarray: Edge detected frame or original frame
        """
        if not self.active:
            return frame
        
        low_threshold = cv2.getTrackbarPos('Low Threshold', window_name)
        high_threshold = cv2.getTrackbarPos('High Threshold', window_name)
        aperture_size = cv2.getTrackbarPos('Aperture Size', window_name)
        
        # Ensure aperture size is odd and in valid range (3, 5, 7)
        aperture_size = max(3, min(7, aperture_size))
        if aperture_size % 2 == 0:
            aperture_size += 1
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold, 
                         apertureSize=aperture_size)
        
        # Convert back to 3-channel for consistent display
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def get_edges_only(self, frame, low_threshold=50, high_threshold=150, aperture_size=3):
        """
        Get edge-detected version of frame with custom parameters.
        
        Args:
            frame (numpy.ndarray): Input frame
            low_threshold (int): Low threshold for edge detection
            high_threshold (int): High threshold for edge detection
            aperture_size (int): Aperture size for Sobel operator
            
        Returns:
            numpy.ndarray: Edge detected frame (single channel)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Ensure aperture size is odd and in valid range
        aperture_size = max(3, min(7, aperture_size))
        if aperture_size % 2 == 0:
            aperture_size += 1
        
        return cv2.Canny(gray, low_threshold, high_threshold, 
                        apertureSize=aperture_size)
    
    def reset(self):
        """Reset edge detection state."""
        self.active = False