"""
Histogram Handler
Handles histogram calculation and visualization.
"""

import cv2
import numpy as np

class HistogramHandler:
    """Handles histogram visualization."""
    
    def __init__(self):
        """Initialize histogram handler."""
        self.show_histogram = False
        self.window_name = 'Histogram'
    
    def toggle(self):
        """Toggle histogram display on/off."""
        self.show_histogram = not self.show_histogram
        if not self.show_histogram:
            self._close_histogram_window()
    
    def is_active(self):
        """
        Check if histogram is being displayed.
        
        Returns:
            bool: True if histogram is active, False otherwise
        """
        return self.show_histogram
    
    def update(self, frame, color_mode):
        """
        Update histogram display based on current frame and color mode.
        
        Args:
            frame (numpy.ndarray): Current frame
            color_mode (str): Current color mode ("COLOR", "GRAY", "HSV")
        """
        if not self.show_histogram:
            self._close_histogram_window()
            return
        
        # Calculate and display histogram
        hist_image = self._calculate_histogram(frame, color_mode)
        cv2.imshow(self.window_name, hist_image)
        
        # Check if window was closed by user
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            self.show_histogram = False
    
    def _calculate_histogram(self, frame, color_mode):
        """
        Calculate histogram for the given frame and color mode.
        
        Args:
            frame (numpy.ndarray): Input frame
            color_mode (str): Color mode ("COLOR", "GRAY", "HSV")
            
        Returns:
            numpy.ndarray: Histogram image
        """
        hist_height, hist_width = 400, 512
        hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
        
        if color_mode == "GRAY":
            return self._draw_grayscale_histogram(frame, hist_image, hist_height, hist_width)
        else:
            return self._draw_color_histogram(frame, hist_image, hist_height, hist_width, color_mode)
    
    def _draw_grayscale_histogram(self, frame, hist_image, hist_height, hist_width):
        """
        Draw histogram for grayscale image.
        
        Args:
            frame (numpy.ndarray): Input frame
            hist_image (numpy.ndarray): Image to draw histogram on
            hist_height (int): Height of histogram image
            hist_width (int): Width of histogram image
            
        Returns:
            numpy.ndarray: Histogram image
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
        
        # Draw histogram
        bin_width = hist_width // 256
        for i in range(1, 256):
            cv2.line(
                hist_image,
                (int((i-1) * bin_width), hist_height - int(hist[i-1])),
                (int(i * bin_width), hist_height - int(hist[i])),
                (255, 255, 255), 2
            )
        
        # Add labels
        cv2.putText(hist_image, 'Grayscale Histogram', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return hist_image
    
    def _draw_color_histogram(self, frame, hist_image, hist_height, hist_width, color_mode):
        """
        Draw histogram for color image.
        
        Args:
            frame (numpy.ndarray): Input frame
            hist_image (numpy.ndarray): Image to draw histogram on
            hist_height (int): Height of histogram image
            hist_width (int): Width of histogram image
            color_mode (str): Color mode ("COLOR" or "HSV")
            
        Returns:
            numpy.ndarray: Histogram image
        """
        # Define colors for different channels
        if color_mode == "HSV":
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # H, S, V
            labels = ['Hue', 'Saturation', 'Value']
            title = 'HSV Histogram'
        else:  # COLOR (BGR)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # B, G, R
            labels = ['Blue', 'Green', 'Red']
            title = 'BGR Histogram'
        
        bin_width = hist_width // 256
        
        # Calculate and draw histogram for each channel
        for channel in range(3):
            hist = cv2.calcHist([frame], [channel], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, hist_height - 50, cv2.NORM_MINMAX)
            
            for i in range(1, 256):
                cv2.line(
                    hist_image,
                    (int((i-1) * bin_width), hist_height - 50 - int(hist[i-1])),
                    (int(i * bin_width), hist_height - 50 - int(hist[i])),
                    colors[channel], 1
                )
        
        # Add title and legend
        cv2.putText(hist_image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add legend
        for i, (color, label) in enumerate(zip(colors, labels)):
            y_pos = hist_height - 30 + (i * 15)
            cv2.putText(hist_image, label, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return hist_image
    
    def _close_histogram_window(self):
        """Close the histogram window if it exists."""
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.window_name)
        except cv2.error:
            # Window doesn't exist, ignore
            pass
    
    def reset(self):
        """Reset histogram state."""
        self.show_histogram = False
        self._close_histogram_window()