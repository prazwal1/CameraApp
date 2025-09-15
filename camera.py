import cv2
import numpy as np

class CameraApp:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.mode = "COLOR"  # Modes: COLOR, GRAY, HSV
        self.alpha = 1.0  # Contrast control (1.0-3.0)
        self.beta = 0     # Brightness control (0-100)
        self.show_hist = False
        self.show_adjust = False
        cv2.namedWindow('Camera')

    def nothing(self, x):
        pass

    def create_trackbars_adjust(self):
        cv2.createTrackbar('Alpha x0.1', 'Camera', int(self.alpha*10), 30, self.nothing)
        cv2.createTrackbar('Beta', 'Camera', self.beta+100, 200, self.nothing)

    def remove_trackbars(self):
        cv2.destroyWindow('Camera')
        cv2.namedWindow('Camera')

    def show_histogram(self, frame):
        h = 300
        w = 512
        hist_img = np.zeros((h, w, 3), dtype=np.uint8)
        if self.mode == "GRAY":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
            for x in range(1, 256):
                cv2.line(hist_img, (x-1, h-int(hist[x-1])), (x, h-int(hist[x])), (255,255,255), 2)
        elif self.mode == "HSV":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            colors = [(255,0,0), (0,255,0), (0,0,255)]
            for i, col in enumerate(colors):
                hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
                for x in range(1, 256):
                    cv2.line(hist_img, (x-1, h-int(hist[x-1])), (x, h-int(hist[x])), col, 2)
        else:  # COLOR
            colors = [(255,0,0), (0,255,0), (0,0,255)]
            for i, col in enumerate(colors):
                hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
                for x in range(1, 256):
                    cv2.line(hist_img, (x-1, h-int(hist[x-1])), (x, h-int(hist[x])), col, 2)
        return hist_img

    def process_color(self, frame):
        return frame

    def process_gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def process_hsv(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def draw_help_text(self, display_frame):
        help_text = "1: Color | 2: Gray | 3: HSV | A: Adjust Brightness and Contrast | H: Histogram"
        quit_text = "Q: Quit"
        mode_text = f"Mode: {self.mode}"
        lines = []

        # Split help_text into multiple lines if too long for frame width
        frame_width = display_frame.shape[1] if display_frame.ndim == 3 else display_frame.shape[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Helper to split text smartly
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

        # Leave some margin
        max_text_width = int(frame_width * 0.95)
        lines.extend(split_text(help_text, max_text_width))
        lines.append(quit_text)
        lines.append(mode_text)
        if self.show_adjust:
            adj_text = f"Alpha: {self.alpha:.1f}  Beta: {self.beta}"
            lines.append(adj_text)

        # Draw each line with vertical spacing
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
            self.show_adjust = not self.show_adjust
        elif key == ord('h'):
            self.show_hist = not self.show_hist
        elif key == ord('q'):
            return False
        return True

    def handle_adjust_mode(self, adjust_initialized):
        if self.show_adjust:
            if not adjust_initialized:
                self.create_trackbars_adjust()
                adjust_initialized = True
            self.alpha = cv2.getTrackbarPos('Alpha x0.1', 'Camera') / 10.0
            self.beta = cv2.getTrackbarPos('Beta', 'Camera') - 100
        else:
            if adjust_initialized:
                self.remove_trackbars()
                adjust_initialized = False
        return adjust_initialized

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
            display_frame = self.process_gray(adj_frame)
        elif self.mode == "HSV":
            display_frame = self.process_hsv(adj_frame)
        else:
            display_frame = self.process_color(adj_frame)
        return display_frame

    def run(self):
        adjust_initialized = False
        running = True
        while running:
            ret, frame = self.cam.read()
            if not ret:
                break

            adjust_initialized = self.handle_adjust_mode(adjust_initialized)
            adj_frame = cv2.convertScaleAbs(frame, alpha=self.alpha, beta=self.beta)

            display_frame = self.handle_running_mode(adj_frame)
            
            self.draw_help_text(display_frame)
            cv2.imshow('Camera', display_frame)

            self.handle_histogram_mode(adj_frame)

            key = cv2.waitKey(1) & 0xFF
            running = self.handle_key(key)

        self.cam.release()
        cv2.destroyAllWindows()
