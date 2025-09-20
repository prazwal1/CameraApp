"""
Line Detection Handler
Handles Hough line detection operations.
"""

import cv2
import numpy as np

class LineDetectionHandler:
    """Handles Hough line detection."""
    
    def __init__(self):
        """Initialize line detection state."""
        self.active = False
    
    def set_active(self, active):
        """
        Set the active state of line detection.
        
        Args:
            active (bool): Whether line detection is active
        """
        self.active = active
    
    def is_active(self):
        """
        Check if line detection is active.
        
        Returns:
            bool: True if active, False otherwise
        """
        return self.active
    
    def create_trackbars(self, window_name, trackbar_manager):
        """
        Create trackbars for Hough line detection parameters.
        
        Args:
            window_name (str): Name of the window
            trackbar_manager: Trackbar manager instance
        """
        trackbar_manager.create_trackbar('Threshold', window_name, 50, 200)
        trackbar_manager.create_trackbar('Min Line Length', window_name, 50, 200)
        trackbar_manager.create_trackbar('Max Line Gap', window_name, 10, 100)
        trackbar_manager.create_trackbar('Canny Low', window_name, 50, 300)
        trackbar_manager.create_trackbar('Canny High', window_name, 150, 300)
    
    def apply(self, frame, window_name):
        """
        Apply Hough line detection if active.
        
        Args:
            frame (numpy.ndarray): Input frame
            window_name (str): Name of the window for trackbar access
            
        Returns:
            numpy.ndarray: Frame with detected lines or original frame
        """
        if not self.active:
            return frame
        
        # Get parameters from trackbars
        threshold = cv2.getTrackbarPos('Threshold', window_name)
        min_line_length = cv2.getTrackbarPos('Min Line Length', window_name)
        max_line_gap = cv2.getTrackbarPos('Max Line Gap', window_name)
        canny_low = cv2.getTrackbarPos('Canny Low', window_name)
        canny_high = cv2.getTrackbarPos('Canny High', window_name)
        
        # Ensure minimum values
        threshold = max(1, threshold)
        min_line_length = max(1, min_line_length)
        max_line_gap = max(1, max_line_gap)
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Ensure 3-channel output
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=threshold,
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        
        # Draw lines on the frame
        result_frame = frame.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add line count text
            line_count = len(lines)
            cv2.putText(result_frame, f'Lines: {line_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_frame
    
    def detect_lines(self, frame, threshold=50, min_line_length=50, max_line_gap=10):
        """
        Detect lines in a frame with custom parameters.
        
        Args:
            frame (numpy.ndarray): Input frame
            threshold (int): Accumulator threshold for line detection
            min_line_length (int): Minimum line length
            max_line_gap (int): Maximum gap between line segments
            
        Returns:
            list: List of detected lines [[x1, y1, x2, y2], ...]
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=threshold,
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        
        return lines if lines is not None else []
    
    def reset(self):
        """Reset line detection state."""
        self.active = False