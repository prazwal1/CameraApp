import yaml
from src.camera_app import CameraApp

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

if __name__ == "__main__":
    config = load_config()

    app = CameraApp(
        # Camera
        camera_index=config.get("camera", {}).get("camera_index", 0),

        # Calibration
        chessboard_size=tuple(config.get("calibration", {}).get("chessboard_size", (9, 6))),
        square_size=config.get("calibration", {}).get("square_size", 15.24),
        target_image_count=config.get("calibration", {}).get("target_image_count", 20),

        # AR
        calibration_file=config.get("ar", {}).get("calibration_file", "output/calibration.npz"),
        model_path=config.get("ar", {}).get("model_path", "models/trex_model.obj"),
        marker_length=config.get("ar", {}).get("marker_length", 0.12),
        model_scale_factor=config.get("ar", {}).get("model_scale_factor", 0.0004),
        rotate_model=config.get("ar", {}).get("rotate_model", True),
    )

    app.run()
