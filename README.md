# Advanced Camera Application

A sophisticated Python OpenCV application for real-time webcam capture with interactive image processing features. This modular application provides various computer vision capabilities including color space conversion, image filtering, edge detection, and line detection.

## 🚀 Features

### Core Functionality
- **Live Webcam Feed** - Real-time video capture from your default camera
- **Modular Architecture** - Well-organized, maintainable code structure
- **Interactive Controls** - Real-time parameter adjustment via trackbars
- **Multiple Processing Modes** - Switch between different image processing techniques

### Color Modes
- **COLOR** - Standard BGR color display
- **GRAY** - Grayscale conversion with histogram support
- **HSV** - HSV color space visualization

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

## 🎮 Keyboard Controls

| Key | Function |
|-----|----------|
| `1` | Switch to Color mode |
| `2` | Switch to Grayscale mode |
| `3` | Switch to HSV mode |
| `A` | Toggle brightness/contrast adjustment |
| `G` | Toggle Gaussian blur filter |
| `B` | Toggle bilateral filter |
| `C` | Toggle Canny edge detection |
| `D` | Toggle Hough line detection |
| `H` | Toggle histogram display |
| `P` | Toggle panorama mode |
| `SPACE` | Panorama capture (when in P mode) |
| `R` | Reset panorama (when in P mode) |
| `X` | Close panorama window (when in P mode) |
| `Q` | Quit the application |

## 📁 Project Structure

```
advanced-camera-app/
├── main.py                     # Entry point
├── src/
│   ├── __init__.py
│   ├── camera_app.py          # Main application 
│   ├── core/
│   │   ├── __init__.py
│   │   └── camera_manager.py   # Camera operations
│   ├── features/
│   │   ├── __init__.py
│   │   ├── color_modes.py      # Color space conversions
│   │   ├── image_adjustments.py # Brightness/contrast
│   │   ├── filters.py          # Gaussian & bilateral 
│   │   ├── edge_detection.py   # Canny edge detection
│   │   ├── line_detection.py   # Hough line detection
│   │   └── histogram.py        # Histogram visualization
│   │   └── panorama.py         # Panorama Mode
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

### Option 1: Using uv (Recommended)

1. **Install [uv](https://github.com/astral-sh/uv):**
   Follow the instructions on the [uv GitHub page](https://github.com/astral-sh/uv) to install `uv` for your platform.

2. **Set up the project:**
   ```bash
   # Navigate to your project directory
   cd advanced-camera-app

   # Create a new virtual environment
   uv venv .venv

   # Activate the virtual environment
   # On Unix/macOS:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate

   # Install dependencies
   uv sync
   ```

3. **Run the application:**
   ```bash
   uv run main.py
   ```

### Option 2: Using pip

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## 🎯 How to Use

1. **Launch the application** using one of the installation methods above
2. **Camera feed** will appear in the main window
3. **Use number keys (1-3)** to switch between color modes
4. **Use letter keys** to toggle different features:
   - Each feature is mutually exclusive (only one active at a time)
   - Trackbars appear when a feature is activated
   - Adjust parameters using the trackbars for real-time effects
5. **Press 'H'** to view histogram in a separate window
6. **Press 'Q'** to quit the application

## ⚙️ Configuration

### Camera Settings
- Default camera index: `0` (can be modified in `CameraManager`)
- Default resolution: `640x480`
- Default FPS: `30`

### Performance Tips
- For better performance on slower systems, consider reducing camera resolution
- Gaussian and Bilateral filters can be computationally intensive with large kernel sizes
- Canny edge detection is generally faster than line detection

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

## 📝 License

AIT License - This project is licensed under the AIT License.

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

- **v1.0.0** - Initial release with full feature set
  - Modular architecture implementation
  - All core image processing features
  - Interactive UI with trackbars
  - Comprehensive documentation

## 📞 Support

For support, questions, or suggestions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the code documentation

---

**Built with ❤️ using Python and OpenCV**