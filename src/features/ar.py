import cv2
import numpy as np
import os

class AugmentedRealityHandler:
    """Handles augmented reality features."""
    
    def __init__(self, calibration_file='output/calibration.npz', model_path='models/trex_model.obj', marker_length=0.05, model_scale_factor=0.0005, rotate_model=True):
        """Initialize AR state."""
        self.active = False
        self.calibration_file = calibration_file
        self.model_path = model_path
        self.marker_length = marker_length  # Marker size in meters
        self.mtx = np.eye(3)
        self.dist = np.zeros((1, 5))
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.model_vertices = None
        self.model_faces = None
        self._load_calibration()
        self.pinhole_params = {'fx': self.mtx[0, 0], 'fy': self.mtx[1, 1], 'cx': self.mtx[0, 2], 'cy': self.mtx[1, 2]}
        self._load_model(model_scale_factor, rotate_model)

    def my_estimatePoseSingleMarkers(self, corners, marker_size, mtx, distortion):
        '''
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        '''
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, -marker_size / 2, 0],
                                  [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        trash = []
        rvecs = []
        tvecs = []
        for c in corners:
            nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)
            trash.append(nada)
        return rvecs, tvecs, trash

    def set_active(self, active):
        """
        Set the active state of AR.
        
        Args:
            active (bool): Whether AR is active
        """
        self.active = active
    
    def is_active(self):
        """Check if AR is active."""
        return self.active

    def _load_calibration(self):
        if os.path.exists(self.calibration_file):
            with np.load(self.calibration_file) as X:
                self.mtx, self.dist = [X[i] for i in ('mtx', 'dist')]
            print("Calibration data loaded.")
        else:
            print(f"WARNING: '{self.calibration_file}' not found. AR and Pinhole modes will not be accurate.")
    
    def _load_model(self, scale_factor=None, rotate=False):
        """Load and preprocess the 3D model."""
        if os.path.exists(self.model_path):
            vertices = []
            faces = []
            with open(self.model_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                                vertices.append(vertex)
                            except ValueError:
                                continue
                    elif line.startswith('f '):
                        parts = line.split()[1:]
                        try:
                            face = []
                            for p in parts:
                                idx = int(p.split('/')[0]) - 1
                                face.append(idx)
                            if len(face) >= 3 and all(0 <= idx < len(vertices) for idx in face):
                                faces.append(face)
                        except (ValueError, IndexError):
                            continue
            
            if vertices and faces:
                self.model_vertices = np.array(vertices, dtype=np.float32)
                self.model_faces = faces
                
                # Center the model by subtracting centroid
                centroid = np.mean(self.model_vertices, axis=0)
                self.model_vertices -= centroid
                
                # Optional rotation to orient upright (e.g., if model is lying on side)
                if rotate:
                    # Example: Rotate 90 degrees around X-axis (adjust as needed)
                    # Rotate 90 degrees around X-axis, then 180 degrees around Z-axis to face towards viewer
                    rotation_x = np.array([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]
                    ])
                    
                    

                    self.model_vertices = np.dot(self.model_vertices, rotation_x.T)

                # Shift so bottom is on the plane (Z=0)
                bb_min = np.min(self.model_vertices, axis=0)
                self.model_vertices[:, 2] -= bb_min[2]
                
                # Auto-scale if no scale_factor provided
                if scale_factor is None:
                    # Compute bounding box max dimension
                    bb_min = np.min(self.model_vertices, axis=0)
                    bb_max = np.max(self.model_vertices, axis=0)
                    bb_size = bb_max - bb_min
                    max_dim = np.max(bb_size)
                    if max_dim > 0:
                        scale_factor = self.marker_length / max_dim * 2.0  # Increased to 200% for larger size
                    else:
                        scale_factor = 1.0
                print(f"Auto scale factor: {scale_factor}")
                # Apply scaling
                self.model_vertices *= scale_factor
                
                print(f"Model loaded, oriented, and scaled by factor {scale_factor}.")
            else:
                print("No valid vertices or faces found in model.")
        else:
            print(f"Model file '{self.model_path}' not found.")
    
    def draw_3d_model(self, frame, window_name):
        # Detect ArUco markers
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        
        if ids is not None and len(ids) > 0:
            # Assume first marker
            marker_corners = corners[0].reshape((4, 2))
            
            # Pose estimation
            rvec, tvec, _ = self.my_estimatePoseSingleMarkers(
                [marker_corners], self.marker_length, self.mtx, self.dist
            )
            
            if self.model_vertices is not None and self.model_faces:
                try:
                    # Project vertices
                    imgpts, _ = cv2.projectPoints(self.model_vertices, rvec[0], tvec[0], self.mtx, self.dist)
                    imgpts = np.int32(imgpts).reshape(-1, 2)
                    
                    # Get frame dimensions for clipping
                    height, width = frame.shape[:2]
                    
                    # Draw faces with clipping
                    for face in self.model_faces:
                        if len(face) >= 3:
                            pts = imgpts[face]
                            # Clip points to frame bounds to prevent crashes from extreme values
                            pts = np.clip(pts, [0, 0], [width - 1, height - 1])
                            # Draw outline (safer than fill)
                            # cv2.fillConvexPoly(frame, pts, (5, 125, 17))  # solid green
                            cv2.polylines(frame, [pts], True, (0, 0, 0), 1)
                except Exception as e:
                    print(f"Error during projection or drawing: {e}")
            else:
                print("Model not loaded.")
        else:
            print("No ArUco marker detected.")
        
        return frame