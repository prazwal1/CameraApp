"""
Panorama Handler
Handles panorama creation by stitching multiple camera frames together.
"""

import cv2
import numpy as np
import time

class PanoramaHandler:
    """Handles panorama creation and image stitching."""
    
    def __init__(self):
        """Initialize panorama handler."""
        self.active = False
        self.capturing = False
        self.frames = []
        self.panorama = None
        self.max_frames = 10
        self.capture_interval = 1.0  # seconds between captures
        self.last_capture_time = 0
        self.current_frame_count = 0
        
        # Feature detector and matcher
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        # Stitching parameters
        self.confidence_threshold = 0.3
        self.reproj_threshold = 4.0
    
    def set_active(self, active):
        """
        Set the active state of panorama mode.
        
        Args:
            active (bool): Whether panorama mode is active
        """
        self.active = active
        if not active:
            self.reset()
    
    def is_active(self):
        """
        Check if panorama mode is active.
        
        Returns:
            bool: True if active, False otherwise
        """
        return self.active
    
    def is_capturing(self):
        """
        Check if currently capturing frames for panorama.
        
        Returns:
            bool: True if capturing, False otherwise
        """
        return self.capturing
    
    def create_trackbars(self, window_name, trackbar_manager):
        """
        Create trackbars for panorama parameters.
        
        Args:
            window_name (str): Name of the window
            trackbar_manager: Trackbar manager instance
        """
        trackbar_manager.create_trackbar('Max Frames', window_name, self.max_frames, 20)
        trackbar_manager.create_trackbar('Interval x10', window_name, int(self.capture_interval * 10), 50)
        trackbar_manager.create_trackbar('Confidence x100', window_name, int(self.confidence_threshold * 100), 100)
        trackbar_manager.create_trackbar('Reproj Thresh', window_name, int(self.reproj_threshold), 10)
    
    def update_from_trackbars(self, window_name):
        """
        Update panorama parameters from trackbars.
        
        Args:
            window_name (str): Name of the window
        """
        if not self.active:
            return
        
        self.max_frames = max(2, cv2.getTrackbarPos('Max Frames', window_name))
        self.capture_interval = cv2.getTrackbarPos('Interval x10', window_name) / 10.0
        self.confidence_threshold = cv2.getTrackbarPos('Confidence x100', window_name) / 100.0
        self.reproj_threshold = max(1, cv2.getTrackbarPos('Reproj Thresh', window_name))
    
    def start_capture(self):
        """Start capturing frames for panorama."""
        if not self.active:
            return
        
        self.capturing = True
        self.frames = []
        self.panorama = None
        self.current_frame_count = 0
        self.last_capture_time = time.time()
        print("Panorama capture started. Move camera slowly left to right or right to left.")
    
    def stop_capture(self):
        """Stop capturing and create panorama."""
        if not self.capturing:
            return
        
        self.capturing = False
        if len(self.frames) >= 2:
            print(f"Creating panorama from {len(self.frames)} frames...")
            self.create_panorama()
        else:
            print("Need at least 2 frames to create panorama.")
    
    def toggle_capture(self):
        """Toggle between start and stop capture."""
        if self.capturing:
            self.stop_capture()
        else:
            self.start_capture()
    
    def process_frame(self, frame):
        """
        Process frame for panorama capture.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Processed frame with status overlay
        """
        if not self.active:
            return frame
        
        current_time = time.time()
        display_frame = frame.copy()
        
        # Auto-capture frames at intervals
        if (self.capturing and 
            current_time - self.last_capture_time >= self.capture_interval and
            len(self.frames) < self.max_frames):
            
            self.capture_frame(frame)
            self.last_capture_time = current_time
        
        # Add status overlay
        self.add_status_overlay(display_frame)
        
        # Show panorama if available
        if self.panorama is not None:
            self.show_panorama()
        
        return display_frame
    
    def capture_frame(self, frame):
        """
        Capture a frame for panorama creation.
        
        Args:
            frame (numpy.ndarray): Frame to capture
        """
        if len(self.frames) >= self.max_frames:
            return
        
        # Store frame
        frame_copy = frame.copy()
        self.frames.append(frame_copy)
        self.current_frame_count += 1
        
        print(f"Captured frame {self.current_frame_count}/{self.max_frames}")
        
        # Auto-stop when max frames reached
        if len(self.frames) >= self.max_frames:
            self.stop_capture()
    
    def create_panorama(self):
        """Create panorama from captured frames."""
        if len(self.frames) < 2:
            print("Need at least 2 frames for panorama creation.")
            return
        
        try:
            # Use OpenCV's Stitcher class for robust panorama creation
            stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
            
            # Configure stitcher parameters
            # stitcher.setPanoConfidenceThresh(self.confidence_threshold)
            
            print("Stitching frames...")
            status, panorama = stitcher.stitch(self.frames)
            
            if status == cv2.Stitcher_OK:
                self.panorama = panorama
                print("Panorama created successfully!")
                self.show_panorama()
            else:
                error_messages = {
                    cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images",
                    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed"
                }
                error_msg = error_messages.get(status, f"Unknown error (code: {status})")
                print(f"Panorama creation failed: {error_msg}")
                
                # Try manual stitching as fallback
                self.create_manual_panorama()
                
        except Exception as e:
            print(f"Error creating panorama: {e}")
            self.create_manual_panorama()
    
    def create_manual_panorama(self):
        """Create panorama using manual feature matching (fallback method)."""
        if len(self.frames) < 2:
            return
        
        try:
            print("Trying manual panorama stitching...")
            
            # Start with the first frame
            result = self.frames[0].copy()
            
            # Stitch each subsequent frame
            for i in range(1, len(self.frames)):
                result = self.stitch_pair(result, self.frames[i])
                if result is None:
                    print(f"Failed to stitch frame {i+1}")
                    return
            
            self.panorama = result
            print("Manual panorama created successfully!")
            self.show_panorama()
            
        except Exception as e:
            print(f"Manual panorama creation failed: {e}")
    
    def stitch_pair(self, img1, img2):
        """
        Stitch two images together.
        
        Args:
            img1 (numpy.ndarray): First image
            img2 (numpy.ndarray): Second image
            
        Returns:
            numpy.ndarray or None: Stitched image or None if failed
        """
        # Convert to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            print("Insufficient features detected")
            return None
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            print("Insufficient good matches found")
            return None
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        homography, mask = cv2.findHomography(
            dst_pts, src_pts, 
            cv2.RANSAC, 
            self.reproj_threshold
        )
        
        if homography is None:
            print("Homography calculation failed")
            return None
        
        # Warp and stitch images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get corners of second image
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners2_warped = cv2.perspectiveTransform(corners2, homography)
        
        # Calculate output size
        all_corners = np.concatenate([
            [[0, 0], [w1, 0], [w1, h1], [0, h1]],
            corners2_warped.reshape(-1, 2)
        ])
        
        x_min, y_min = np.min(all_corners, axis=0).astype(int)
        x_max, y_max = np.max(all_corners, axis=0).astype(int)
        
        # Adjust homography for translation
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        homography_adjusted = translation @ homography
        
        # Create output canvas
        output_width = x_max - x_min
        output_height = y_max - y_min
        result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Warp second image
        warped = cv2.warpPerspective(img2, homography_adjusted, (output_width, output_height))
        
        # Place first image
        y_offset = -y_min
        x_offset = -x_min
        result[y_offset:y_offset+h1, x_offset:x_offset+w1] = img1
        
        # Blend images
        mask = (warped > 0).any(axis=2)
        result[mask] = warped[mask]
        
        return result
    
    def show_panorama(self):
        """Display the created panorama in a separate window."""
        if self.panorama is None:
            return
        
        # Resize panorama for display if too large
        display_panorama = self.panorama.copy()
        height, width = display_panorama.shape[:2]
        
        max_display_width = 1200
        if width > max_display_width:
            scale = max_display_width / width
            new_width = max_display_width
            new_height = int(height * scale)
            display_panorama = cv2.resize(display_panorama, (new_width, new_height))
        
        cv2.imshow('Panorama', display_panorama)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            cv2.destroyWindow('Panorama')
            self.reset()
    
    def save_panorama(self, filename=None):
        """
        Save the current panorama to file.
        
        Args:
            filename (str, optional): Filename to save to
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if self.panorama is None:
            print("No panorama to save")
            return False
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"panorama_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, self.panorama)
            print(f"Panorama saved as {filename}")
            return True
        except Exception as e:
            print(f"Error saving panorama: {e}")
            return False
    
    def add_status_overlay(self, frame):
        """
        Add status information overlay to the frame.
        
        Args:
            frame (numpy.ndarray): Frame to add overlay to
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Status text
        if self.capturing:
            status = f"Capturing: {len(self.frames)}/{self.max_frames}"
            color = (0, 255, 0)  # Green
        else:
            status = "Ready (Press SPACE to start/stop capture)"
            color = (0, 255, 255)  # Yellow
        
        # Add background rectangle
        text_size = cv2.getTextSize(status, font, font_scale, thickness)[0]
        cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, status, (15, 30), font, font_scale, color, thickness)
        
        # Add instructions
        instructions = [
            "P: Toggle panorama mode",
            "SPACE: Start/Stop capture", 
            "S: Save panorama",
            "R: Reset"
        ]
        
        y_offset = 60
        for instruction in instructions:
            cv2.putText(frame, instruction, (15, y_offset), font, 0.4, (200, 200, 200), 1)
            y_offset += 20
    
    def handle_key(self, key):
        """
        Handle keyboard input for panorama mode.
        
        Args:
            key (int): Key code
            
        Returns:
            bool: True if key was handled, False otherwise
        """
        if not self.active:
            return False
        
        if key == ord(' '):  # Space - start/stop capture
            self.toggle_capture()
            return True
        elif key == ord('s'):  # Save panorama
            self.save_panorama()
            return True
        elif key == ord('r'):  # Reset
            self.reset()
            return True
        
        return False
    
    def reset(self):
        """Reset panorama state."""
        self.capturing = False
        self.frames = []
        self.panorama = None
        self.current_frame_count = 0
        self.last_capture_time = 0
        
        # Close panorama window if open
        try:
            cv2.destroyWindow('Panorama')
            print("Panorama reset.")
        except:
            pass
        
    
    def get_frame_count(self):
        """
        Get the current number of captured frames.
        
        Returns:
            int: Number of captured frames
        """
        return len(self.frames)
    
    def has_panorama(self):
        """
        Check if a panorama has been created.
        
        Returns:
            bool: True if panorama exists, False otherwise
        """
        return self.panorama is not None