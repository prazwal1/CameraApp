"""
Trackbar Manager
Manages OpenCV trackbars creation and cleanup.
"""

import cv2

class TrackbarManager:
    """Manages trackbar operations for OpenCV windows."""
    
    def __init__(self):
        """Initialize trackbar manager."""
        self.active_trackbars = {}  # window_name -> [trackbar_names]
    
    def create_trackbar(self, trackbar_name, window_name, initial_value, max_value, callback=None):
        """
        Create a trackbar in the specified window.
        
        Args:
            trackbar_name (str): Name of the trackbar
            window_name (str): Name of the window
            initial_value (int): Initial value of the trackbar
            max_value (int): Maximum value of the trackbar
            callback (callable, optional): Callback function for trackbar changes
        """
        if callback is None:
            callback = self._nothing
        
        cv2.createTrackbar(trackbar_name, window_name, initial_value, max_value, callback)
        
        # Track active trackbars
        if window_name not in self.active_trackbars:
            self.active_trackbars[window_name] = []
        self.active_trackbars[window_name].append(trackbar_name)
    
    def remove_all_trackbars(self, window_name):
        """
        Remove all trackbars from a window by recreating it.
        
        Args:
            window_name (str): Name of the window
        """
        try:
            # Check if window exists
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(window_name)
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            
            # Clear tracking
            if window_name in self.active_trackbars:
                self.active_trackbars[window_name] = []
                
        except cv2.error:
            # Window doesn't exist, just clear tracking
            if window_name in self.active_trackbars:
                self.active_trackbars[window_name] = []
    
    def get_trackbar_value(self, trackbar_name, window_name):
        """
        Get the current value of a trackbar.
        
        Args:
            trackbar_name (str): Name of the trackbar
            window_name (str): Name of the window
            
        Returns:
            int: Current trackbar value, or 0 if trackbar doesn't exist
        """
        try:
            return cv2.getTrackbarPos(trackbar_name, window_name)
        except cv2.error:
            return 0
    
    def set_trackbar_value(self, trackbar_name, window_name, value):
        """
        Set the value of a trackbar.
        
        Args:
            trackbar_name (str): Name of the trackbar
            window_name (str): Name of the window
            value (int): Value to set
        """
        try:
            cv2.setTrackbarPos(trackbar_name, window_name, value)
        except cv2.error:
            pass  # Trackbar doesn't exist
    
    def trackbar_exists(self, trackbar_name, window_name):
        """
        Check if a trackbar exists.
        
        Args:
            trackbar_name (str): Name of the trackbar
            window_name (str): Name of the window
            
        Returns:
            bool: True if trackbar exists, False otherwise
        """
        return (window_name in self.active_trackbars and 
                trackbar_name in self.active_trackbars[window_name])
    
    def get_active_trackbars(self, window_name):
        """
        Get list of active trackbars for a window.
        
        Args:
            window_name (str): Name of the window
            
        Returns:
            list: List of trackbar names
        """
        return self.active_trackbars.get(window_name, [])
    
    def cleanup(self):
        """Clean up all tracked resources."""
        self.active_trackbars = {}
    
    def _nothing(self, x):
        """Default callback function for trackbars."""
        pass