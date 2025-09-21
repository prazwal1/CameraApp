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
        else:
            print("Calibration deactivated.")
    
    def is_active(self):
        """Check if calibration is active."""
        return self.active

    def capture_frame(self, frame):
        """
        Capture a frame for calibration if active.
        """
        if not self.is_active() or self.image_captured >= self.target_image_count:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if ret and (time.time() - self.last_capture_time > 1):  # 1 sec delay
            cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)
            self.objpoints.append(self._object_points())
            self.imgpoints.append(corners)
            self.image_captured += 1
            self.last_capture_time = time.time()
            print(f"[INFO] Captured image {self.image_captured}/{self.target_image_count}")
     
    def _compute_errors(self):
        self.mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], self.camera_matrix, self.dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            self.mean_error += error
        print(f"Total re-projection error: {self.mean_error/len(self.objpoints):.4f}")

    def run(self, frame):
        """
        Process a frame for calibration if active.
        """
        self.capture_frame(frame)

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
            except cv2.error as e:
                print(f"Calibration failed: {e}")
        return frame