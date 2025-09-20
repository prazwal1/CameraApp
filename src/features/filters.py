"""
Filter Handler
Handles image filtering operations (Gaussian blur, Bilateral filter).
"""

import cv2

class FilterHandler:
    """Handles various image filters."""
    
    def __init__(self):
        """Initialize filter states."""
        self.gaussian_active = False
        self.bilateral_active = False
    
    def set_gaussian_active(self, active):
        """
        Set the active state of Gaussian blur.
        
        Args:
            active (bool): Whether Gaussian blur is active
        """
        self.gaussian_active = active
    
    def set_bilateral_active(self, active):
        """
        Set the active state of bilateral filter.
        
        Args:
            active (bool): Whether bilateral filter is active
        """
        self.bilateral_active = active
    
    def create_gaussian_trackbars(self, window_name, trackbar_manager):
        """
        Create trackbars for Gaussian blur parameters.
        
        Args:
            window_name (str): Name of the window
            trackbar_manager: Trackbar manager instance
        """
        trackbar_manager.create_trackbar('Kernel Size', window_name, 1, 20)
        trackbar_manager.create_trackbar('Sigma X', window_name, 0, 100)
    
    def create_bilateral_trackbars(self, window_name, trackbar_manager):
        """
        Create trackbars for bilateral filter parameters.
        
        Args:
            window_name (str): Name of the window
            trackbar_manager: Trackbar manager instance
        """
        trackbar_manager.create_trackbar('Diameter', window_name, 9, 20)
        trackbar_manager.create_trackbar('Sigma Color', window_name, 75, 150)
        trackbar_manager.create_trackbar('Sigma Space', window_name, 75, 150)
    
    def apply_gaussian(self, frame, window_name):
        """
        Apply Gaussian blur if active.
        
        Args:
            frame (numpy.ndarray): Input frame
            window_name (str): Name of the window for trackbar access
            
        Returns:
            numpy.ndarray: Filtered frame
        """
        if not self.gaussian_active:
            return frame
        
        kernel_size = cv2.getTrackbarPos('Kernel Size', window_name)
        sigma_x = cv2.getTrackbarPos('Sigma X', window_name)
        
        # Ensure kernel size is odd and at least 1
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply Gaussian blur
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma_x)
    
    def apply_bilateral(self, frame, window_name):
        """
        Apply bilateral filter if active.
        
        Args:
            frame (numpy.ndarray): Input frame
            window_name (str): Name of the window for trackbar access
            
        Returns:
            numpy.ndarray: Filtered frame
        """
        if not self.bilateral_active:
            return frame
        
        diameter = cv2.getTrackbarPos('Diameter', window_name)
        sigma_color = cv2.getTrackbarPos('Sigma Color', window_name)
        sigma_space = cv2.getTrackbarPos('Sigma Space', window_name)
        
        # Ensure diameter is at least 1
        diameter = max(1, diameter)
        
        # Apply bilateral filter
        return cv2.bilateralFilter(frame, diameter, sigma_color, sigma_space)
    
    def is_gaussian_active(self):
        """
        Check if Gaussian blur is active.
        
        Returns:
            bool: True if active, False otherwise
        """
        return self.gaussian_active
    
    def is_bilateral_active(self):
        """
        Check if bilateral filter is active.
        
        Returns:
            bool: True if active, False otherwise
        """
        return self.bilateral_active
    
    def reset(self):
        """Reset all filter states."""
        self.gaussian_active = False
        self.bilateral_active = False