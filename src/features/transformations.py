import cv2

class ImageTransformationHandler:
    """Handles image transformations like rotation, translation and scaling."""
    def __init__(self):
        """Initialize with default transformation values."""
        self.angle = 0          # Rotation angle in degrees
        self.scale = 1.0        # Scaling factor
        self.tx = 0             # Translation in x direction
        self.ty = 0             # Translation in y direction
        self.active = False
        
    def set_active(self, active):
        """
        Set the active state of image transformations.
        
        Args:
            active (bool): Whether transformations are active
        """
        self.active = active
    
    def is_active(self):
        """
        Check if image transformations are active.
        
        Returns:
            bool: True if active, False otherwise
        """
        return self.active

    def create_trackbars(self, window_name, trackbar_manager):
        """
        Create trackbars for rotation, scaling and translation.
        
        Args:
            window_name (str): Name of the window
            trackbar_manager (TrackbarManager): The trackbar manager instance
        """
        trackbar_manager.create_trackbar(
            'Rotation', window_name, int(self.angle), 360
        )
        trackbar_manager.create_trackbar(
            'Scale', window_name, int(self.scale * 100), 300
        )
        trackbar_manager.create_trackbar(
            'Translate X', window_name, self.tx + 50, 100
        )
        trackbar_manager.create_trackbar(
            'Translate Y', window_name, self.ty + 50, 100
        )

    def update_from_trackbars(self, window_name):
        """
        Update transformation values from trackbars.
        
        Args:
            window_name (str): Name of the window
        """
        if not self.active:
            return
        
        self.angle = cv2.getTrackbarPos("Rotation", window_name)
        self.scale = cv2.getTrackbarPos("Scale", window_name) / 100.0
        self.tx = cv2.getTrackbarPos("Translate X", window_name) - 50  # Centering translation
        self.ty = cv2.getTrackbarPos("Translate Y", window_name) - 50  # Centering translation 
    
    def apply(self, frame, window_name):
        """
        Apply transformations to the frame if active.
        
        Args:
            frame (numpy.ndarray): Input frame
        """
        if not self.active:
            return frame
        
        self.update_from_trackbars(window_name)

        # Get the transformation matrix
        M = cv2.getRotationMatrix2D((frame.shape[1] // 2, frame.shape[0] // 2), self.angle, self.scale)
        M[0, 2] += self.tx
        M[1, 2] += self.ty

        # Apply the transformation
        transformed = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        return transformed