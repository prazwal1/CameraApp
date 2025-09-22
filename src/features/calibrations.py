import cv2
import numpy as np
import time
import os

class CameraCalibration:
    """Perform camera calibration using chessboard images."""
    def __init__(self, chessboard_size=(9, 6), square_size=25.4, target_image_count=20):
        """
        Initialize the camera calibration parameters.
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.target_image_count = target_image_count
        self.calibrated = False
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_captured = 0
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.active = False
        self.last_capture_time = 0

        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
    def _object_points(self):
        """Prepare object points based on the chessboard size and square size."""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp 
    
    def set_active(self, active):
        """
        Set calibration active state.
        """
        self.active = active
        if active:
            print("Calibration started. Capture chessboard images...")
            print("Instructions: Hold the chessboard in view at different angles, positions, and orientations.")
            print("The system will automatically capture images when a stable chessboard is detected.")
            print("Aim for varied poses to improve calibration accuracy.")
        else:
            print("Calibration deactivated.")
            if self.image_captured > 0 and self.image_captured < self.target_image_count:
                print(f"Warning: Only {self.image_captured} images captured. For best results, capture at least {self.target_image_count}.")
    
    def is_active(self):
        """Check if calibration is active."""
        return self.active

    def capture_frame(self, frame):
        """
        Capture a frame for calibration if active.
        """
        if not self.is_active() or self.image_captured >= self.target_image_count:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if ret:
            # Refine corner positions for accuracy
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            
            # Always draw detected corners for visual feedback
            cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)
            
            # Capture only if enough time has passed (to allow user to reposition)
            if time.time() - self.last_capture_time > 2:  # Increased to 2 sec for better user control
                self.objpoints.append(self._object_points())
                self.imgpoints.append(corners)
                self.image_captured += 1
                self.last_capture_time = time.time()
                print(f"[INFO] Captured image {self.image_captured}/{self.target_image_count}")
                # Flash or indicate capture
                cv2.putText(frame, "Captured!", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        return frame
     
    def _compute_errors(self):
        self.mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            self.mean_error += error
        mean_error = self.mean_error / len(self.objpoints)
        print(f"Total re-projection error: {mean_error:.4f}")
        if mean_error > 1.0:
            print("Warning: High re-projection error. Consider recapturing images for better calibration.")

    def run(self, frame):
        """
        Process a frame for calibration if active.
        """
        if self.active:
            frame = self.capture_frame(frame)
            
            # Add progress overlay
            progress_text = f"Captured: {self.image_captured}/{self.target_image_count}"
            cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if self.image_captured < self.target_image_count:
                instruction_text = "Move chessboard to new position"
                cv2.putText(frame, instruction_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.image_captured >= self.target_image_count and not self.calibrated:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
                    self.objpoints, self.imgpoints, gray.shape[::-1], None, None
                )
                if self.ret:
                    self.calibrated = True
                    self.active = False
                    print("Calibration successful.")
                    self._compute_errors()
                    np.savez(
                        "output/calibration.npz",
                        mtx=self.camera_matrix, dist=self.dist_coeffs,
                        rvecs=self.rvecs, tvecs=self.tvecs
                    )
                    print("Calibration data saved to output/calibration.npz")
                    # Show success on frame
                    cv2.putText(frame, "Calibration Complete!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print("Calibration failed: Insufficient data or poor image quality.")
            except cv2.error as e:
                print(f"Calibration failed: {e}")
                # Reset for retry
                self.objpoints = []
                self.imgpoints = []
                self.image_captured = 0
        return frame