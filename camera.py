import cv2
import numpy as np

class CameraApp:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.mode = "COLOR"  # Modes: COLOR, GRAY, HSV
        self.alpha = 1.0  # Contrast control (1.0-3.0)
        self.beta = 0     # Brightness control (0-100)

        # Flags
        self.show_hist = False
        self.show_adjust = False
        self.show_gaussian = False
        self.show_bilateral = False
        self.show_canny = False
        self.hough_transform = False
        self.mode_panorama = False

        # State
        self.active_trackbar_mode = None   # "ADJUST", "GAUSSIAN", "BILATERAL", or None

        cv2.namedWindow('Camera')

    def nothing(self, x):
        pass

    def create_trackbars_hough_lines(self):
        cv2.createTrackbar('Threshold', 'Camera', 50, 200, self.nothing)
        cv2.createTrackbar('Min Line Length', 'Camera', 50, 200, self.nothing)
        cv2.createTrackbar('Max Line Gap', 'Camera', 10, 100, self.nothing)
        

    def create_trackbars_canny(self):
        cv2.createTrackbar('Threshold1', 'Camera', 50, 500, self.nothing)
        cv2.createTrackbar('Threshold2', 'Camera', 150, 500, self.nothing)


    def create_trackbars_adjust(self):
        cv2.createTrackbar('Alpha x0.1', 'Camera', int(self.alpha*10), 30, self.nothing)
        cv2.createTrackbar('Beta', 'Camera', self.beta+100, 200, self.nothing)

    def create_trackbars_gaussian(self):
        cv2.createTrackbar('Kernel Size', 'Camera', 1, 20, self.nothing) 
        cv2.createTrackbar('SigmaX', 'Camera', 0, 100, self.nothing)

    def create_trackbars_bilateral(self):
        cv2.createTrackbar('Diameter', 'Camera', 1, 20, self.nothing) 
        cv2.createTrackbar('SigmaColor', 'Camera', 0, 100, self.nothing)
        cv2.createTrackbar('SigmaSpace', 'Camera', 0, 100, self.nothing)

    def remove_trackbars(self):
        """Destroy the window and recreate it to remove all trackbars."""
        cv2.destroyWindow('Camera')
        cv2.namedWindow('Camera')
        self.active_trackbar_mode = None

    def show_histogram(self, frame):
        h, w = 300, 512
        hist_img = np.zeros((h, w, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        num_channels = 1 if self.mode == "GRAY" else 3

        for i in range(num_channels):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
            color = colors[i] if num_channels == 3 else (255, 255, 255)
            for x in range(1, 256):
                cv2.line(hist_img, 
                         (x - 1, h - int(hist[x - 1])),
                         (x, h - int(hist[x])),
                         color, 2)
        return hist_img

    def process_color(self, frame):
        return frame

    def process_gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def process_hsv(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def draw_help_text(self, display_frame):
        help_text = "1: Color | 2: Gray | 3: HSV | A: Adjust | H: Histogram | G: Gaussian Blur | B: Bilateral | C: Canny Edge | D: Hough Lines"
        quit_text = "Q: Quit"
        mode_text = f"Mode: {self.mode}"
        lines = []

        frame_width = display_frame.shape[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        def split_text(text, max_width):
            words = text.split(' ')
            lines = []
            current = ""
            for word in words:
                test = current + (' ' if current else '') + word
                size = cv2.getTextSize(test, font, font_scale, thickness)[0][0]
                if size > max_width and current:
                    lines.append(current)
                    current = word
                else:
                    current = test
            if current:
                lines.append(current)
            return lines

        max_text_width = int(frame_width * 0.95)
        lines.extend(split_text(help_text, max_text_width))
        lines.append(quit_text)
        lines.append(mode_text)
        if self.show_adjust:
            adj_text = f"Alpha: {self.alpha:.1f}  Beta: {self.beta}"
            lines.append(adj_text)

        y = 30
        for i, text in enumerate(lines):
            if i == len(lines) - 1:
                color = (255, 0, 0)
            elif i == len(lines) - 2:
                color = (0, 0, 255)
            else:
                color = (113, 179, 60)
            cv2.putText(display_frame, text, (10, y), font, font_scale, color, thickness)
            y += 30

    def handle_key(self, key):
        if key == ord('1'):
            self.mode = "COLOR"
        elif key == ord('2'):
            self.mode = "GRAY"
        elif key == ord('3'):
            self.mode = "HSV"
        elif key == ord('a'):
            self.toggle_mode("ADJUST")
        elif key == ord('g'):
            self.toggle_mode("GAUSSIAN")
        elif key == ord('b'):
            self.toggle_mode("BILATERAL")
        elif key == ord('h'):
            self.show_hist = not self.show_hist
        elif key == ord('c'):
            self.toggle_mode("CANNY")
        elif key == ord('d'):
            self.toggle_mode("HOUGH")
        elif key == ord('q'):
            return False
        return True

    def handle_canny_mode(self, frame):
        if self.show_canny and self.active_trackbar_mode == "CANNY":
            t1 = cv2.getTrackbarPos('Threshold1', 'Camera')
            t2 = cv2.getTrackbarPos('Threshold2', 'Camera')
            return cv2.Canny(frame, t1, t2)
        return frame
    


    def toggle_mode(self, mode):
        """Enable one mode and disable others (mutual exclusivity)."""
        if self.active_trackbar_mode == mode:
            # Turn off
            self.remove_trackbars()
            if mode == "ADJUST": self.show_adjust = False
            if mode == "GAUSSIAN": self.show_gaussian = False
            if mode == "BILATERAL": self.show_bilateral = False
        else:
            # Reset everything
            self.remove_trackbars()
            self.show_adjust = self.show_gaussian = self.show_bilateral = False

            if mode == "ADJUST":
                self.create_trackbars_adjust()
                self.show_adjust = True
            elif mode == "GAUSSIAN":
                self.create_trackbars_gaussian()
                self.show_gaussian = True
            elif mode == "BILATERAL":
                self.create_trackbars_bilateral()
                self.show_bilateral = True
            elif mode == "CANNY":
                self.create_trackbars_canny()
                self.show_canny = True
            elif mode == "HOUGH":
                self.create_trackbars_hough_lines()
                self.hough_transform = True
            self.active_trackbar_mode = mode

    def handle_adjust_mode(self):
        if self.show_adjust and self.active_trackbar_mode == "ADJUST":
            self.alpha = cv2.getTrackbarPos('Alpha x0.1', 'Camera') / 10.0
            self.beta = cv2.getTrackbarPos('Beta', 'Camera') - 100

    def handle_gaussian_mode(self, frame):
        if self.show_gaussian and self.active_trackbar_mode == "GAUSSIAN":
            ksize = cv2.getTrackbarPos('Kernel Size', 'Camera')
            sigmaX = cv2.getTrackbarPos('SigmaX', 'Camera')
            if ksize % 2 == 0: ksize += 1
            if ksize < 1: ksize = 1
            return cv2.GaussianBlur(frame, (ksize, ksize), sigmaX)
        return frame

    def handle_bilateral_mode(self, frame):
        if self.show_bilateral and self.active_trackbar_mode == "BILATERAL":
            diameter = cv2.getTrackbarPos('Diameter', 'Camera')
            sigmaColor = cv2.getTrackbarPos('SigmaColor', 'Camera')
            sigmaSpace = cv2.getTrackbarPos('SigmaSpace', 'Camera')
            if diameter < 1: diameter = 1
            return cv2.bilateralFilter(frame, diameter, sigmaColor, sigmaSpace)
        return frame

    def handle_hough_lines(self, frame):
        if self.hough_transform and self.active_trackbar_mode == "HOUGH":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            threshold = cv2.getTrackbarPos('Threshold', 'Camera')
            min_line_length = cv2.getTrackbarPos('Min Line Length', 'Camera')
            max_line_gap = cv2.getTrackbarPos('Max Line Gap', 'Camera')
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame


    def handle_histogram_mode(self, adj_frame):
        if self.show_hist:
            hist_img = self.show_histogram(adj_frame)
            cv2.imshow('Histogram', hist_img)
            if cv2.getWindowProperty('Histogram', cv2.WND_PROP_VISIBLE) < 1:
                self.show_hist = False
        else:
            if cv2.getWindowProperty('Histogram', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow('Histogram')

    def handle_running_mode(self, adj_frame):
        if self.mode == "GRAY":
            return self.process_gray(adj_frame)
        elif self.mode == "HSV":
            return self.process_hsv(adj_frame)
        return self.process_color(adj_frame)

    def run(self):
        running = True
        while running:
            ret, frame = self.cam.read()
            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF
            running = self.handle_key(key)

            self.handle_adjust_mode()
            adj_frame = cv2.convertScaleAbs(frame, alpha=self.alpha, beta=self.beta)
            adj_frame = self.handle_gaussian_mode(adj_frame)
            adj_frame = self.handle_bilateral_mode(adj_frame)
            adj_frame = self.handle_canny_mode(adj_frame)
            adj_frame = self.handle_hough_lines(adj_frame)
            display_frame = self.handle_running_mode(adj_frame)
            self.draw_help_text(display_frame)
            cv2.imshow('Camera', display_frame)

            self.handle_histogram_mode(adj_frame)

        self.cam.release()
        cv2.destroyAllWindows()
