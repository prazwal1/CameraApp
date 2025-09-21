"""
Main Camera Application Class
Orchestrates all camera features and handles the main application loop.
"""

import cv2
from .core.camera_manager import CameraManager
from .features.color_modes import ColorModeHandler
from .features.image_adjustments import ImageAdjustmentHandler
from .features.filters import FilterHandler
from .features.edge_detection import EdgeDetectionHandler
from .features.line_detection import LineDetectionHandler
from .features.panorama import PanoramaHandler
from .features.histogram import HistogramHandler
from .ui.trackbar_manager import TrackbarManager
from .ui.display_manager import DisplayManager
from .ui.keyboard_handler import KeyboardHandler

class CameraApp:
    """Main camera application orchestrating all features."""
    
    def __init__(self):
        """Initialize the camera application with all components."""
        # Core components
        self.camera = CameraManager()
        self.trackbar_manager = TrackbarManager()
        self.display = DisplayManager()
        self.keyboard = KeyboardHandler()
        
        # Feature handlers
        self.color_modes = ColorModeHandler()
        self.adjustments = ImageAdjustmentHandler()
        self.filters = FilterHandler()
        self.edge_detection = EdgeDetectionHandler()
        self.line_detection = LineDetectionHandler()
        self.histogram = HistogramHandler()
        self.panorama = PanoramaHandler()
        # Application state
        self.running = True
        self.active_feature = None
        
        # Setup UI
        self._setup_ui()
        self._setup_keyboard_bindings()
    
    def _setup_ui(self):
        """Initialize the main UI window."""
        cv2.namedWindow('Advanced Camera App', cv2.WINDOW_AUTOSIZE)
    
    def _setup_keyboard_bindings(self):
        """Setup keyboard event handlers."""
        self.keyboard.bind_key('1', lambda: self.color_modes.set_mode("COLOR"))
        self.keyboard.bind_key('2', lambda: self.color_modes.set_mode("GRAY"))
        self.keyboard.bind_key('3', lambda: self.color_modes.set_mode("HSV"))
        self.keyboard.bind_key('a', lambda: self._toggle_feature("adjustments"))
        self.keyboard.bind_key('g', lambda: self._toggle_feature("gaussian"))
        self.keyboard.bind_key('b', lambda: self._toggle_feature("bilateral"))
        self.keyboard.bind_key('c', lambda: self._toggle_feature("canny"))
        self.keyboard.bind_key('d', lambda: self._toggle_feature("hough"))
        self.keyboard.bind_key('p', lambda: self._toggle_feature("panorama"))
        self.keyboard.bind_key(' ', lambda: self._handle_panorama_keys(ord(' ')))
        self.keyboard.bind_key('s', lambda: self._handle_panorama_keys(ord('s')))
        self.keyboard.bind_key('r', lambda: self._handle_panorama_keys(ord('r')))
        self.keyboard.bind_key('h', lambda: self.histogram.toggle())
        self.keyboard.bind_key('q', lambda: self._quit())
    
    def _handle_panorama_keys(self, key):
        """Handle panorama-specific keyboard inputs."""
        self.panorama.handle_key(key)

    
    def _toggle_feature(self, feature_name):
        """Toggle a specific feature on/off."""
        if self.active_feature == feature_name:
            # Turn off current feature
            self.trackbar_manager.remove_all_trackbars('Advanced Camera App')
            self.active_feature = None
            self._reset_feature_states()
        else:
            # Turn on new feature
            self.trackbar_manager.remove_all_trackbars('Advanced Camera App')
            self._reset_feature_states()
            self.active_feature = feature_name
            self._activate_feature(feature_name)
    
    def _reset_feature_states(self):
        """Reset all feature states."""
        self.adjustments.set_active(False)
        self.filters.set_gaussian_active(False)
        self.filters.set_bilateral_active(False)
        self.edge_detection.set_active(False)
        self.line_detection.set_active(False)
        self.panorama.set_active(False)

    
    def _activate_feature(self, feature_name):
        """Activate a specific feature and setup its trackbars."""
        if feature_name == "adjustments":
            self.adjustments.set_active(True)
            self.adjustments.create_trackbars('Advanced Camera App', self.trackbar_manager)
        elif feature_name == "gaussian":
            self.filters.set_gaussian_active(True)
            self.filters.create_gaussian_trackbars('Advanced Camera App', self.trackbar_manager)
        elif feature_name == "bilateral":
            self.filters.set_bilateral_active(True)
            self.filters.create_bilateral_trackbars('Advanced Camera App', self.trackbar_manager)
        elif feature_name == "canny":
            self.edge_detection.set_active(True)
            self.edge_detection.create_trackbars('Advanced Camera App', self.trackbar_manager)
        elif feature_name == "hough":
            self.line_detection.set_active(True)
            self.line_detection.create_trackbars('Advanced Camera App', self.trackbar_manager)
        elif feature_name == "panorama":
            self.panorama.set_active(True)
            self.panorama.create_trackbars('Advanced Camera App', self.trackbar_manager)
    
    def _quit(self):
        """Quit the application."""
        self.running = False
    
    def _process_frame(self, frame):
        """Process frame through all active features."""
        processed_frame = frame.copy()
        
        # Apply adjustments
        if self.adjustments.is_active():
            self.adjustments.update_from_trackbars('Advanced Camera App')
            processed_frame = self.adjustments.apply(processed_frame)
        
        
        if self.panorama.is_active():
            self.panorama.update_from_trackbars('Advanced Camera App')
            processed_frame = self.panorama.process_frame(processed_frame)
        else:
            processed_frame = self.panorama.process_frame(processed_frame)
        
        # Apply filters
        processed_frame = self.filters.apply_gaussian(processed_frame, 'Advanced Camera App')
        processed_frame = self.filters.apply_bilateral(processed_frame, 'Advanced Camera App')
        
        # Apply edge detection
        processed_frame = self.edge_detection.apply(processed_frame, 'Advanced Camera App')
        
        # Apply line detection
        processed_frame = self.line_detection.apply(processed_frame, 'Advanced Camera App')
        
        # Apply color mode
        processed_frame = self.color_modes.apply(processed_frame)
        
        return processed_frame
    
    def run(self):
        """Main application loop."""
        print("Starting Advanced Camera Application...")
        print("Press 'H' to see help overlay")
        
        while self.running:
            # Capture frame
            frame = self.camera.get_frame()
            if frame is None:
                print("Failed to capture frame. Exiting...")
                break
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self.keyboard.handle_key(key)
            
            # Process frame
            processed_frame = self._process_frame(frame)
            
            # Add help overlay
            display_frame = self.display.add_help_overlay(
                processed_frame,
                self.color_modes.current_mode,
                self.active_feature,
                self.adjustments.alpha,
                self.adjustments.beta
            )
            
            # Show main window
            cv2.imshow('Advanced Camera App', display_frame)
            
            # Handle histogram
            self.histogram.update(processed_frame, self.color_modes.current_mode)
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        print("Application closed.")