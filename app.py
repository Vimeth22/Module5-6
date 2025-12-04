from flask import Flask, render_template, Response, request
import cv2

# Import our strategy classes
from tracking_strategies import ArucoStrategy, CSRTStrategy, SAM2Strategy

app = Flask(__name__)


class CameraContext:
    def __init__(self):
        # Open default camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Could not open default camera (index 0).")

        # Start with ArUco (marker) strategy
        self.strategy = ArucoStrategy()
        self.frame_counter = 0

    def switch_strategy(self, mode: str):
        """Switch between marker / markerless / sam2 strategies."""
        self.frame_counter = 0  # Reset when switching
        mode = (mode or "").strip().lower()

        if mode == "marker":
            self.strategy = ArucoStrategy()
        elif mode == "markerless":
            self.strategy = CSRTStrategy()
        elif mode == "sam2":
            self.strategy = SAM2Strategy()
        else:
            print(f"Unknown mode '{mode}', defaulting to marker (Aruco).")
            self.strategy = ArucoStrategy()

        print(f"Switched to strategy: {mode}")

    def get_feed(self):
        """Generator that yields JPEG frames for MJPEG streaming."""
        while True:
            if not self.cap.isOpened():
                # Try to reopen if something went wrong
                self.cap.open(0)
                if not self.cap.isOpened():
                    print("ERROR: Camera is not available.")
                    break

            ret, frame = self.cap.read()
            if not ret:
                print("WARNING: Failed to read frame from camera.")
                break

            # Process frame with current strategy
            processed = self.strategy.update(frame, self.frame_counter)
            self.frame_counter += 1

            # Encode to JPEG
            success, buffer = cv2.imencode(".jpg", processed)
            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )


# Global camera context
cam_context = CameraContext()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        cam_context.get_feed(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/select_mode", methods=["POST"])
def select_mode():
    mode = request.form.get("mode")
    cam_context.switch_strategy(mode)
    return "OK", 200


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  
    app.run(host="0.0.0.0", port=port, debug=False)
