# Advanced Camera Application

A sophisticated Python OpenCV application for real-time webcam capture with interactive image processing features. This modular application provides a wide range of computer vision capabilities including color space conversion, image filtering, edge detection, line detection, panorama creation, geometric transformations, camera calibration, and augmented reality overlays.

## 🚀 Features

### Core Functionality
- **Live Webcam Feed** – Real-time video capture from your default camera
- **Modular Architecture** – Well-organized, maintainable code structure
- **Interactive Controls** – Real-time parameter adjustment via trackbars
- **Multiple Processing Modes** – Switch between different image processing techniques

### Color Modes
- **COLOR** – Standard BGR color display
- **GRAY** – Grayscale conversion with histogram support
- **HSV** – HSV color space visualization

### Image Enhancement
- **Brightness & Contrast Adjustment**
  - Real-time adjustment using trackbars
  - Independent control of contrast (Alpha) and brightness (Beta)
  - Live preview with current values displayed

### Filtering Options
- **Gaussian Blur**
  - Adjustable kernel size and sigma values
  - Effective noise reduction and smoothing
- **Bilateral Filter**
  - Edge-preserving noise reduction
  - Configurable diameter, sigma color, and sigma space parameters

### Edge & Line Detection
- **Canny Edge Detection**
  - Adjustable low and high thresholds
  - Configurable aperture size
  - Real-time edge visualization
- **Hough Line Detection**
  - Probabilistic Hough Transform implementation
  - Adjustable detection parameters
  - Live line count display

### Visualization Tools
- **Histogram Display**
  - Real-time histogram calculation
  - Support for all color modes (BGR, Grayscale, HSV)
  - Separate window with color-coded channels
  - Automatic scaling and normalization

### Panorama Mode
- **Panorama Capture**
  - Capture multiple frames to create a panorama
  - Adjustable capture interval and maximum frames
  - Real-time preview of the stitched panorama
  - Options to reset and save the panorama

### Transformation Mode
- **Geometric Transformations**
  - Real-time rotation, translation, and scaling
  - Adjustable angle, translation, and scale factor
  - Live preview with current values displayed

### Camera Calibration
- **Calibration Mode**
  - Activate camera calibration routines
  - Reset calibration with a key press

### Augmented Reality
- **AR Mode**
  - Overlay 3D models onto the camera feed in real time

## 🎮 Keyboard Controls

| Key      | Function                                 |
|----------|------------------------------------------|
| `1`      | Switch to Color mode                     |
| `2`      | Switch to Grayscale mode                 |
| `3`      | Switch to HSV mode                       |
| `A`      | Toggle brightness/contrast adjustment    |
| `G`      | Toggle Gaussian blur filter              |
| `B`      | Toggle bilateral filter                  |
| `C`      | Toggle Canny edge detection              |
| `D`      | Toggle Hough line detection              |
| `H`      | Toggle histogram display                 |
| `P`      | Toggle panorama mode                     |
| `T`      | Toggle transformation mode               |
| `K`      | Toggle camera calibration mode           |
| `F`      | Toggle augmented reality mode            |
| `SPACE`  | Panorama capture (when in P mode)        |
| `S`      | Save panorama (when in P mode)           |
| `R`      | Reset panorama (when in P mode)          |
| `X`      | Reset calibration                        |
| `F1` | Toggle help window                       |
| `Q`      | Quit the application                     |

## 📁 Project Structure

```
advanced-camera-app/
├── main.py                     # Entry point
├── src/
│   ├── __init__.py
│   ├── camera_app.py           # Main application
│   ├── core/
│   │   ├── __init__.py
│   │   └── camera_manager.py   # Camera operations
│   ├── features/
│   │   ├── __init__.py
│   │   ├── color_modes.py      # Color space conversions
│   │   ├── image_adjustments.py# Brightness/contrast
│   │   ├── filters.py          # Gaussian & bilateral
│   │   ├── edge_detection.py   # Canny edge detection
│   │   ├── line_detection.py   # Hough line detection
│   │   ├── histogram.py        # Histogram visualization
│   │   ├── panorama.py         # Panorama Mode
│   │   ├── transformations.py  # Geometric transformations
│   │   ├── calibrations.py     # Camera calibration
│   │   └── ar.py               # Augmented reality
│   └── ui/
│       ├── __init__.py
│       ├── trackbar_manager.py # Trackbar operations
│       ├── display_manager.py  # UI overlays and display
│       └── keyboard_handler.py # Keyboard input handling
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8 or higher
- Working webcam
- OpenCV-compatible system

### 1. Clone the Repository

```bash
git clone https://github.com/prazwal1/CameraApp.git
cd CameraApp
```

### 2. Using uv (Recommended)

```bash
uv venv .venv
# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

uv sync
uv run main.py
```

### 3. Using pip

```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

pip install -r requirements.txt
python main.py
```

## 🎯 How to Use

1. **Launch the application** using one of the installation methods above.
2. **Camera feed** will appear in the main window.
3. **Use number keys (1-3)** to switch between color modes.
4. **Use letter keys** to toggle different features:
   - Only one feature is active at a time (mutually exclusive).
   - Trackbars appear when a feature is activated.
   - Adjust parameters using the trackbars for real-time effects.
5. **Press 'H'** to view histogram in a separate window.
6. **Press 'F1'** to toggle the help window.
7. **Press 'Q'** to quit the application.

## ⚙️ Configuration
The application uses a YAML configuration file (`config.yaml`) to manage camera, calibration, and AR settings. Example:

```yaml
camera:
  camera_index: 0   # Default camera index

calibration:
  chessboard_size: [9, 6]   # Chessboard pattern size (rows, cols)
  square_size: 15.24        # Size of each square in mm
  target_image_count: 20    # Number of images required for calibration

ar:
  calibration_file: "output/calibration.npz"   # Path to saved calibration file
  model_path: "models/trex_model.obj"          # 3D model path
  marker_length: 0.12                          # Marker size in meters
  model_scale_factor: 0.0004                   # Scaling factor for 3D model
  rotate_model: true                           # Apply rotation correction
```

- Edit `config.yaml` to customize camera index, calibration parameters, and AR model settings.
- The application loads these settings at startup for flexible configuration.

### Camera Settings
- Default camera index: `0` (can be modified in `CameraManager`)
- Default resolution: `640x480`
- Default FPS: `30`

### Performance Tips
- For better performance on slower systems, consider reducing camera resolution.
- Gaussian and Bilateral filters can be computationally intensive with large kernel sizes.
- Canny edge detection is generally faster than line detection.

## 🏗️ Architecture Overview

The application follows a modular design pattern:

- **Core Layer**: Camera management and basic operations
- **Features Layer**: Individual image processing capabilities
- **UI Layer**: User interface components and input handling
- **Main Application**: Orchestrates all components

Each feature is implemented as an independent module, making it easy to:
- Add new features
- Modify existing functionality
- Test individual components
- Maintain code organization

## 🔧 Development

### Adding New Features

1. Create a new module in `src/features/`
2. Implement the feature following the existing pattern:
   - `set_active()` method
   - `create_trackbars()` method if needed
   - `apply()` method for processing
3. Register the feature in `camera_app.py`
4. Add keyboard binding in `_setup_keyboard_bindings()`

### Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Include comprehensive docstrings
- Maintain modular design principles

## 🐛 Troubleshooting

### Common Issues

**Camera not found:**
- Ensure your webcam is connected and not in use by other applications
- Try changing the camera index in `CameraManager.__init__()`

**Poor performance:**
- Reduce camera resolution
- Use smaller kernel sizes for filters
- Close other resource-intensive applications

**Trackbars not responding:**
- Ensure the feature is active (press the corresponding key)
- Check that the window has focus

**Application crashes:**
- Check that all dependencies are properly installed
- Ensure Python version compatibility (3.8+)
- Review console output for error messages


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 System Requirements

- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 2GB minimum (4GB recommended)
- **Camera**: USB webcam or built-in camera
- **Display**: 1024x768 minimum resolution

## 🔄 Version History

- **v1.0.0** – Initial release with full feature set
  - Modular architecture implementation
  - All core image processing features
  - Interactive UI with trackbars
  - Comprehensive documentation

## 📞 Support

For support, questions, or suggestions:
- Open an issue on GitHub
- Contact via email: st125976@ait.asia

---

**Built with ❤️ using Python and OpenCV**