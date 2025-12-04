
# Module5-6 

This module implements a **real-time object tracking framework** that supports **dynamic switching between tracking algorithms**.  
The system integrates Flask, OpenCV, ArUco detection, CSRT tracking, and SAM-based segmentation.

---

# Module5-6 part 1 (the pdf of the answer)

The answer for this is in the **Module 5-6 .pdf**

---

## Folder Structure

```
MODULE5-6/
│── app.py                           # Main Flask server, video feed, and mode switching
│── README.md
│── tracking_strategies.py           # All tracking strategy classes (Aruco, CSRT, SAM2)
│
├── __pycache__/
├── static/                          # Stores assets/output if needed
└── templates/                       # Frontend UI for viewing the stream and selecting modes and HTML templates for Flask

```

---

# Module Overview

Module 6 introducing a **modular, pluggable tracking framework**.  
It allows you to switch between tracking algorithms LIVE without restarting the server.

Users can choose from:

1. **Aruco Marker Tracking** – Fiducial marker detection  
2. **CSRT Tracker** – Marker-less object tracking  
3. **SAM2 Tracking** – Segmentation-based tracking using precomputed SAM masks  

This enables comparison of accuracy, robustness, and performance between fundamentally different tracking approaches.

---

# app.py 

### Responsibilities
- Initializes the webcam stream  
- Manages global tracking context  
- Switches algorithms via `/select_mode`  
- Streams processed frames to the browser  
- Delegates per-frame processing to the active strategy  

### Important Components
- `CameraContext` — Holds:
  - `cap` : webcam feed  
  - `strategy` : currently selected tracking strategy  
  - `frame_counter` : used for SAM mask indexing  

- `/video_feed` — Returns the live MJPEG stream  
- `/select_mode` — POST endpoint to switch algorithms dynamically  

---

# tracking_strategies.py (Tracking Algorithms)

This file defines the **Strategy Pattern** used in Module 6.

Module includes **three classes**, each encapsulating one tracking method:

---

## Part 1 ArUcoStrategy – Marker-Based Tracking
- Uses `cv2.aruco` dictionary  
- Detects markers in each frame  
- Draws bounding boxes + marker IDs  
- Very stable, high accuracy  
- Best for controlled environments  

---

## Part 2 CSRTStrategy – Markerless Object Tracking
- Uses OpenCV’s CSRT tracker  
- User selects ROI on first frame  
- Tracker automatically follows object  
- Great for real-world objects  
- More robust than KCF, MIL  

---

## Part 3 SAM2Strategy – Segmentation-Based Tracking
- Loads precomputed SAM masks (from segmentation.npz)  
- Uses mask contours to outline objects  
- Frame-aligned segmentation tracking  
- Good for shape-based tracking  

---

# index.html (Frontend UI)

This file controls the **web interface**.  
It includes:

### Real-time Live Stream  
Displayed inside:

```
<img src="{{ url_for('video_feed') }}">
```

### Algorithm Selection Buttons  
Each button triggers:

```js
fetch('/select_mode', { method: 'POST', body: formData })
```

### Supported Modes
- **marker** → ArUco  
- **markerless** → CSRT  
- **sam2** → SAM mask tracking  

### Clean UI Layout
- Video preview panel  
- Right-side control panel  
- Buttons visually highlight active mode  

---

# How to Run

### 1. Install dependencies

```bash
pip install flask opencv-python numpy
```

If ArUco dictionary errors occur:

```bash
pip install opencv-contrib-python
```

### 2. Start Flask server

```bash
python app.py
```

### 3. Open in browser

```
http://127.0.0.1:5000
```

### 4. Switch tracking modes
- Once you go to the web app you can see 3 parts in blue tabs.
- You can click each tab for part 1-3 and check the results. 

---

# Notes & Requirements

- A working webcam is required  
- To make the segmentation.npz file you have to write a code and get the marks. 
  - The code is here which is `segmentation.ipynb`. (you have to run this code in google colab and get the segmentation.npz file)
  - In the google colab you have to upload the demo video inside the google colab.
- SAM2 strategy needs a `segmentation.npz` file containing:
  - `masks`
  - `frame_indices`
- CSRT strategy requires user selection of an ROI  
- ArUco requires good lighting for detection  


