# Hand Gesture-Based Brightness Control

## Overview

This project allows users to control their screen brightness using hand gestures detected via a webcam. It utilizes OpenCV, MediaPipe, and the `screen-brightness-control` library to detect hand landmarks and adjust brightness accordingly. Additionally, the program stops execution when a smile is detected using Haar cascades.

## Features

- Adjusts screen brightness based on the distance between thumb and index finger.
- Locks brightness if hand position remains stable for 2 seconds.
- Unlocks brightness using a thumbs-up gesture.
- Stops execution upon detecting a smile.

## Requirements

Make sure you have the following dependencies installed before running the program:

```sh
pip install opencv-python numpy mediapipe screen-brightness-control comtypes
```

## Usage

1. Run the script:
   ```sh
   python script.py
   ```
2. Place your hand in front of the camera.
3. Adjust brightness by changing the distance between thumb and index finger.
4. Hold hand position for 2 seconds to lock brightness.
5. Show a thumbs-up gesture to unlock brightness.
6. Smile at the camera to stop the program.

## File Structure

```
├── script.py           # Main script for brightness control
├── README.md           # Project documentation
```

## How It Works

- **Hand Tracking:** Uses MediaPipe to detect hand landmarks.
- **Distance Calculation:** Computes the distance between the thumb and index finger.
- **Brightness Control:** Maps the hand distance to brightness levels (0-100%).
- **Lock Mechanism:** Locks brightness if the hand is stable for 2 seconds.
- **Unlock Mechanism:** Detects a thumbs-up gesture to unlock brightness.
- **Smile Detection:** Uses Haar cascades to detect a smile and exit the program.

## License

This project is for educational purposes. Modify and use it as needed!
