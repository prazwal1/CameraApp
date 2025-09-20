"""
Image Adjustment Handler
Handles brightness and contrast adjustments.
"""

import cv2

class ImageAdjustmentHandler:
    """Handles brightness and contrast adjustments."""
    
    def __init__(self):
        """Initialize with default adjustment values."""
        self.alpha = 1.0  # Contrast control (1.0-3.0)
        self.beta = 0     # Brightness control (-100 to +100)
        self.active = False
    
    def set_active(self, active):
        """
        Set the active state of image adjustments.
        
        Args:
            active (bool): Whether adjustments are active
        """
        self.active = active
    
    def is_active(self):
        """
        Check if image adjustments are active.
        
        Returns:
            bool: True if active, False otherwise
        """
        return self.active
    
    def create_trackbars(self, window_name, trackbar_manager):
        """
        Create trackbars for brightness and contrast adjustment.
        
        Args:
            window_name (str): Name of the window
            trackbar_manager: Trackbar manager instance
        """
        trackbar_manager.create_trackbar(
            'Contrast x10', window_name, int(self.alpha * 10), 30
        )
        trackbar_manager.create_trackbar(
            'Brightness', window_name, self.beta + 100, 200
        )
    
    def update_from_trackbars(self, window_name):
        """
        Update adjustment values from trackbars.
        
        Args:
            window_name (str): Name of the window
        """
        if not self.active:
            return
        
        self.alpha = cv2.getTrackbarPos('Contrast x10', window_name) / 10.0
        self.beta = cv2.getTrackbarPos('Brightness', window_name) - 100
        
        # Ensure alpha is at least 0.1 to avoid completely dark images
        self.alpha = max(0.1, self.alpha)
    
    def apply(self, frame):
        """
        Apply brightness and contrast adjustments to a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Adjusted frame
        """
        if not self.active:
            return frame
        
        return cv2.convertScaleAbs(frame, alpha=self.alpha, beta=self.beta)
    
    def set_values(self, alpha, beta):
        """
        Manually set adjustment values.
        
        Args:
            alpha (float): Contrast multiplier (0.1-3.0)
            beta (int): Brightness offset (-100 to +100)
        """
        self.alpha = max(0.1, min(3.0, alpha))
        self.beta = max(-100, min(100, beta))
    
    def reset(self):
        """Reset adjustments to default values."""
        self.alpha = 1.0
        self.beta = 0
    
    def get_values(self):
        """
        Get current adjustment values.
        
        Returns:
            tuple: (alpha, beta) values
        """
        return (self.alpha, self.beta)