import cv2
import numpy as np
from abc import ABC, abstractmethod
import os

# --- Abstract Base Class ---
class TrackerStrategy(ABC):
    @abstractmethod
    def update(self, frame, frame_count):
        """Process the frame and return the annotated frame."""
        pass


# --- Strategy 1: Marker Tracker (ArUco) ---
class ArucoStrategy(TrackerStrategy):
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 
        self.params.adaptiveThreshConstant = 5.0 
        self.params.adaptiveThreshWinSizeMax = 35 
        self.params.minMarkerPerimeterRate = 0.02 

    def update(self, frame, frame_count):
        if frame is None:
            return frame

        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self.params)

        if ids is not None:
            centers = []
            for i, corner_set in enumerate(corners):
                pts = corner_set[0].astype(np.int32)
                cv2.polylines(frame, [pts], True, (57, 255, 20), 3)
                c_x = int(np.mean(pts[:, 0]))
                c_y = int(np.mean(pts[:, 1]))
                centers.append((c_x, c_y))
                cv2.circle(frame, (c_x, c_y), 5, (0, 0, 255), -1)
                marker_id = ids[i][0]
                cv2.putText(frame, f"ID: {marker_id}", (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (57, 255, 20), 2)
            
            if len(centers) >= 2:
                for i in range(len(centers) - 1):
                    cv2.line(frame, centers[i], centers[i + 1], (0, 255, 0), 3)

        cv2.putText(frame, "Mode: ArUco Marker Tracking (Improved)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (57, 255, 20), 2)
        return frame


# --- Strategy 2: Marker-less Tracker (CSRT) ---
class CSRTStrategy(TrackerStrategy):
    def __init__(self):
        self.tracker = None
        self.initialized = False
        self.bbox_color = (255, 0, 255)

    def _create_tracker(self):
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        elif hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        else:
            raise RuntimeError("CSRT tracker not available. Install opencv-contrib-python.")

    def init_tracker(self, frame):
        h, w = frame.shape[:2]
        roi_size = 100
        start_box = (w // 2 - roi_size // 2, h // 2 - roi_size // 2, roi_size, roi_size)
        self.tracker = self._create_tracker()
        self.tracker.init(frame, start_box)
        self.initialized = True
        print("CSRT Tracker Initialized at center.")

    def update(self, frame, frame_count):
        if frame is None:
            return frame

        if not self.initialized:
            h, w = frame.shape[:2]
            s = 100
            p1 = (w // 2 - s // 2, h // 2 - s // 2)
            p2 = (w // 2 + s // 2, h // 2 + s // 2)
            cv2.rectangle(frame, p1, p2, (200, 200, 200), 2)
            if frame_count > 30: 
                self.init_tracker(frame)
            cv2.putText(frame, "Mode: CSRT (Marker-less)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.bbox_color, 2)
            return frame

        success, box = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.bbox_color, 3)
            cv2.putText(frame, "Target Locked", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.bbox_color, 2)
        else:
            cv2.putText(frame, "Tracking Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, "Mode: CSRT (Marker-less)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.bbox_color, 2)
        return frame


# --- Strategy 3: SAM2 Segmentation (Offline Masks) ---
class SAM2Strategy(TrackerStrategy):
    def __init__(self):
        self.masks = None
        self.num_masks = 0
        self.video_cap = None
        self.frame_idx = -1
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.video_path = os.path.join(base_dir, "static", "sam2Demo.mp4")
        self.mask_path = os.path.join(base_dir, "static", "segmentation.npz")

        print(f"DEBUG: SAM2 looking for video at: {self.video_path}")
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.mask_path):
            print(f"ERROR: NPZ file not found at {self.mask_path}. You must create this file.")
            return

        try:
            # FIX: Ensure allow_pickle=True is present
            data = np.load(self.mask_path, allow_pickle=True)
            
            # Robustly find the masks array
            if "masks" in data:
                self.masks = data["masks"]
            elif len(data.files) == 1:
                self.masks = data[data.files[0]]
                
            if isinstance(self.masks, np.ndarray) and self.masks.ndim >= 2:
                if self.masks.dtype == object:
                    self.masks = np.array(list(self.masks))
                    
                if self.masks.ndim >= 2:
                    self.num_masks = self.masks.shape[0]
                    print(f"SUCCESS: Loaded {self.num_masks} SAM2 masks with shape {self.masks.shape}.")
                else:
                    self.masks = None
            else:
                self.masks = None

        except Exception as e:
            print(f"CRITICAL ERROR loading NPZ: {e}. **The file is corrupted/empty.**")
        
        # Load video
        if os.path.exists(self.video_path):
            self.video_cap = cv2.VideoCapture(self.video_path)
        
    def _get_next_video_frame(self):
        if self.video_cap is None or not self.video_cap.isOpened():
            return None
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_cap.read()
            if not ret:
                return None
        return frame

    def _get_next_mask(self):
        if self.masks is None or self.num_masks == 0:
            return None

        self.frame_idx = (self.frame_idx + 1) % self.num_masks
        mask = self.masks[self.frame_idx]

        # Robustly reduce mask dimensions to (H, W)
        if mask.ndim >= 3:
            mask = np.max(mask, axis=tuple(range(mask.ndim - 2))) 
        
        # Convert to a binary mask (0 or 255)
        mask = (mask > 0).astype(np.uint8) * 255  
        return mask

    def update(self, frame, frame_count):
        
        video_frame = self._get_next_video_frame()

        if video_frame is None:
            if frame is not None:
                cv2.putText(frame, "SAM2 Video Error - Check Paths", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame
            return None

        current_mask = self._get_next_mask()

        if current_mask is not None:
            
            # Resize mask to video size if needed
            if current_mask.shape[:2] != video_frame.shape[:2]:
                current_mask = cv2.resize(current_mask, (video_frame.shape[1], video_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Find all external contours
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # --- KEY FIX: Draw ONLY the largest contour (the ball) ---
            if contours:
                # Find the contour with the maximum area
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Use a low threshold (10 pixels) to filter out small noise
                MIN_AREA_THRESHOLD = 10 
                
                if cv2.contourArea(largest_contour) > MIN_AREA_THRESHOLD: 
                    # Draw ONLY the largest contour (Cyan color: 255, 255, 0)
                    cv2.drawContours(video_frame, [largest_contour], -1, (255, 255, 0), 3)
            # --------------------------------------------------------
                    
        cv2.putText(video_frame, "Mode: SAM2 (Offline Playback)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return video_frame

# --- Main Application Loop ---
# (Keep this function as you provided it, as it handles the key presses and setup)
def main_loop():
    # ... (Your existing main_loop code)
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("\n--- WARNING: Camera not found. Testing SAM2 functionality only. ---")
        placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8) 
    else:
        placeholder_frame = None
        print("\n--- Camera Active ---")

    aruco_tracker = ArucoStrategy()
    csrt_tracker = CSRTStrategy()
    sam2_tracker = SAM2Strategy()
    
    current_strategy = aruco_tracker 
    frame_count = 0
    
    print("\n--- Strategy Ready ---")
    print("Press 'A' for ArUco, 'C' for CSRT, 'S' for SAM2. Press 'Q' to quit.")

    while True:
        if current_strategy != sam2_tracker and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = placeholder_frame 

        annotated_frame = current_strategy.update(frame, frame_count)

        if annotated_frame is not None:
            cv2.imshow("Tracking Demo", annotated_frame)

        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            current_strategy = aruco_tracker
            print("Switched to strategy: aruco")
        elif key == ord('c'):
            current_strategy = csrt_tracker
            csrt_tracker.initialized = False
            print("Switched to strategy: csrt")
        elif key == ord('s'):
            current_strategy = sam2_tracker
            print("Switched to strategy: sam2")

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()