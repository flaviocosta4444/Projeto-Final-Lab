# Exercise Pose Analysis System

This application uses computer vision to analyze human poses during exercises and provides real-time feedback with a 3D model reference.

## Features

- Real-time pose detection using MediaPipe
- Split-screen interface showing user camera feed and 3D model reference
- 3D model shown in side view for better visualization
- Real-time scoring and posture suggestions
- Countdown timer between exercise stages
- Support for multiple exercises: Squat, Push-up, Plank, Lunge, Deadlift, and Shoulder Press

## Requirements

- Python 3.7 or higher
- Webcam
- Dependencies listed in requirements.txt

## Installation

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```

2. Select an exercise from the user interface
3. Follow the 3D model shown on the right side of the screen
4. Try to match the posture shown by the model to get a high score
5. Watch for feedback and suggestions to improve your form

## Input Folder

To use the image analysis feature, place your exercise images in the `input` folder at the root of the project.

## Controls

- Press 'q' to exit the exercise view and return to the main menu 