import cv2
import numpy as np

class DisplayManager:
    """Manages display overlays and UI elements."""
    
    def __init__(self):
        """Initialize display manager."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        self.line_spacing = 25
        self.help_window_name = "Help Controls"
        self.help_window_created = False
    
    def add_help_overlay(self, frame, color_mode, active_feature, alpha, beta):
        """
        Display help text in a separate window.
        
        Args:
            frame (numpy.ndarray): Input frame (used for sizing reference)
            color_mode (str): Current color mode
            active_feature (str): Currently active feature
            alpha (float): Current contrast value
            beta (int): Current brightness value
            
        Returns:
            numpy.ndarray: Original frame (unchanged)
        """
        # Create help text lines
        help_lines = self._create_help_text(color_mode, active_feature, alpha, beta)
        
        # Calculate window dimensions
        overlay_width = 600
        overlay_height = len(help_lines) * self.line_spacing + 40
        
        # Create help window if not already created
        if not self.help_window_created:
            cv2.namedWindow(self.help_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.help_window_name, overlay_width, overlay_height)
            self.help_window_created = True
        
        # Create help window content
        help_frame = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
        help_frame[:] = (30, 30, 30)  # Dark gray background
        
        # Draw help text
        self._draw_help_text(help_frame, help_lines, 10, 20)
        
        # Display in separate window
        cv2.imshow(self.help_window_name, help_frame)
        
        return frame  # Return original frame unchanged
    
    def _create_help_text(self, color_mode, active_feature, alpha, beta):
        """
        Create help text lines based on current state.
        
        Args:
            color_mode (str): Current color mode
            active_feature (str): Currently active feature
            alpha (float): Current contrast value
            beta (int): Current brightness value
            
        Returns:
            list: List of (text, color) tuples
        """
        lines = [
            ("Keyboard Controls:", (255, 255, 255)),
            ("1: Color | 2: Gray | 3: HSV", (100, 255, 100)),
            ("A: Adjust | G: Gaussian | B: Bilateral", (100, 255, 100)),
            ("C: Canny Edge | D: Hough Lines", (100, 255, 100)),
            ("P: Panorama | T: Transformations", (100, 255, 100)),
            ("K: Calibration | F: Augmented Reality", (100, 255, 100)),
            ("H: Histogram | Q: Quit", (100, 255, 100)),
            ("", (255, 255, 255)),  # Empty line
            (f"Current Mode: {color_mode}", (255, 200, 100)),
        ]
        
        if active_feature:
            feature_names = {
                "adjustments": "Brightness/Contrast",
                "gaussian": "Gaussian Blur",
                "bilateral": "Bilateral Filter",
                "canny": "Canny Edge Detection",
                "hough": "Hough Line Detection",
                "panorama": "Panorama Mode"
            }
            feature_display = feature_names.get(active_feature, active_feature.title())
            lines.append((f"Active Feature: {feature_display}", (100, 200, 255)))
        
        if active_feature == "adjustments":
            lines.append((f"Contrast: {alpha:.1f} | Brightness: {beta}", (200, 200, 255)))
        
        if active_feature == "panorama":
            lines.append(("", (255, 255, 255)))  # Empty line
            lines.append(("Panorama Controls:", (255, 255, 255)))
            lines.append(("SPACE: Capture frame or start/stop", (100, 255, 100)))
            lines.append(("S: Stitch panorama", (100, 255, 100)))
            lines.append(("R: Reset", (100, 255, 100)))
            
        
        return lines
    
    def _draw_help_text(self, frame, help_lines, x_start, y_start):
        """
        Draw help text lines on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            help_lines (list): List of (text, color) tuples
            x_start (int): Starting x position
            y_start (int): Starting y position
        """
        current_y = y_start
        
        for text, color in help_lines:
            if text:  # Skip empty lines for spacing
                cv2.putText(
                    frame, text, (x_start, current_y),
                    self.font, self.font_scale, color, self.thickness
                )
            current_y += self.line_spacing
    
    def add_status_text(self, frame, text, position="top-right", color=(0, 255, 0)):
        """
        Add status text to frame at specified position.
        
        Args:
            frame (numpy.ndarray): Input frame
            text (str): Text to display
            position (str): Position ("top-left", "top-right", "bottom-left", "bottom-right")
            color (tuple): BGR color tuple
            
        Returns:
            numpy.ndarray: Frame with status text
        """
        frame_height, frame_width = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
        
        # Calculate position
        if position == "top-left":
            x, y = 10, 30
        elif position == "top-right":
            x, y = frame_width - text_size[0] - 10, 30
        elif position == "bottom-left":
            x, y = 10, frame_height - 10
        elif position == "bottom-right":
            x, y = frame_width - text_size[0] - 10, frame_height - 10
        else:
            x, y = 10, 30  # Default to top-left
        
        # Add background rectangle for better readability
        padding = 5
        cv2.rectangle(
            frame,
            (x - padding, y - text_size[1] - padding),
            (x + text_size[0] + padding, y + padding),
            (0, 0, 0), -1
        )
        
        # Add text
        cv2.putText(frame, text, (x, y), self.font, self.font_scale, color, self.thickness)
        
        return frame
    
    def create_info_panel(self, width, height, title, info_dict):
        """
        Create an information panel with key-value pairs.
        
        Args:
            width (int): Panel width
            height (int): Panel height
            title (str): Panel title
            info_dict (dict): Dictionary of key-value pairs to display
            
        Returns:
            numpy.ndarray: Info panel image
        """
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add title
        title_y = 30
        cv2.putText(panel, title, (10, title_y), self.font, 0.8, (255, 255, 255), 2)
        
        # Add separator line
        cv2.line(panel, (10, title_y + 10), (width - 10, title_y + 10), (100, 100, 100), 1)
        
        # Add info items
        current_y = title_y + 40
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(panel, text, (10, current_y), self.font, self.font_scale, (200, 200, 200), 1)
            current_y += self.line_spacing
            
            if current_y > height - 10:  # Prevent overflow
                break
        
        return panel
    
    def destroy_help_window(self):
        """Destroy the help window if it exists."""
        if self.help_window_created:
            cv2.destroyWindow(self.help_window_name)
            self.help_window_created = False