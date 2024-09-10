# Distracted Driver Detection System

## Overview

This project implements a **distracted driver detection system** using **YOLOv8**, **OpenCV**, and **Pygame**. The goal is to identify various distracted driving behaviors from video input in real-time. When a risky behavior (like phone usage, eating, etc.) is detected, the system triggers a visual and audio alert to warn of the dangerous activity.

## Features

- **Real-time Driver Detection**: Uses YOLO for detecting distracted driving behaviors from video.
- **Bounding Boxes and Confidence Scores**: Displays bounding boxes around detected objects, along with the class and confidence score.
- **Custom Text Overlay**: Adds informative text labels with confidence levels and class names.
- **Audio Alarm**: Plays an alarm sound when distracted behaviors are detected.
- **User-friendly Interface**: Real-time video feed with overlaid information.

## Project Structure

```
.
├── alarm.wav                # Alarm sound to notify when distraction is detected
├── BestDriver.pt            # Pre-trained YOLO model for driver behavior detection
├── video1.mp4               # Test video to run detection on
├── distracted_drivers.py    # Main Python script containing the detection code
└── README.md                # Project documentation
```

## Prerequisites

Before running this project, make sure you have the following dependencies installed:

- Python 3.x
- OpenCV: `pip install opencv-python`
- Ultralytics YOLO: `pip install ultralytics`
- Pygame: `pip install pygame`

## How to Run

1. Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/MohammedHamza0/DistractedDriverDetection.git
cd DistractedDriverDetection
```

2. Install the required Python packages:


3. Place your YOLO model (`BestDriver.pt`) in the project directory.

4. Place the test video (`video1.mp4`) in the project directory or use your own video for detection.

5. Run the script:

```bash
python distracted_drivers.py
```

6. The video will open in a new window. Detected classes (e.g., distracted drivers) will be marked with bounding boxes. If distracted behavior is detected, an alarm sound will play.

Press **Esc** to stop the video and exit the program.

## Code Explanation

### Main Detection Loop

The core of the program processes video frames, runs YOLO predictions, and performs the following:

- Draws bounding boxes around detected driver behaviors.
- Adds text overlays with class names and confidence scores.
- Plays an alarm sound if risky behavior is detected.
- Displays the processed frames in a window.

### Text Overlay

The `draw_text_with_background` function draws text on the video frame with a background and border to make the labels easier to read.

### Audio Alert

If a distracted driving behavior is detected, an audio alarm (`alarm.wav`) is triggered using the **Pygame** library.

## Example Output

When running the program, you will see a window displaying the video, with bounding boxes and class names. Distracted driving behaviors will have a red bounding box and trigger the alarm, while safe behaviors will have a green box.

## Future Improvements

- **Model Optimization**: Improve the accuracy by training a more specific model or fine-tuning on a custom dataset.
- **Multi-object Detection**: Expand detection to handle multiple drivers or detect non-driver distractions in the vehicle.
- **Live Camera Feed**: Integrate a live feed from a camera for real-time detection in vehicles.
