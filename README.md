# Face Recognition Attendance System

## Description
This Python project implements a real-time face recognition attendance system. It uses the **face_recognition** library and **OpenCV** to capture video from a webcam, detect faces, and recognize them by comparing against a dataset of known images (each labeled with a person’s name). When a face is recognized, the system logs the person’s name and timestamp into an attendance CSV file for the current date. The goal is to automate attendance tracking by identifying individuals in real time.

## Features
- Real-time face detection and recognition via webcam.  
- Maintains a dataset of known faces (images with names).  
- Automatically logs attendance to a dated CSV file (one file per day).  
- Easily add new individuals by placing a labeled image in the dataset folder.  
- No manual roll call needed for attendance.  

## Installation
Use `pip` to install the required libraries:

```bash
pip install face_recognition opencv-python
pip install numpy
```
## Usage
- Create a folder named Images/ in the project directory.
- Add clear images of each person in Images/, naming each image file as the person’s name (e.g., Alice.jpg).
- The webcam will activate. Ensure faces are clearly visible to the camera.
- When a known face is detected, the person’s name and time will be recorded in a CSV file named with the current date (e.g., 2025-05-02.csv).
- Press 'q' in the video window to quit the application.

## Requirements
- Python 3.x
- face_recognition
- opencv-python
- numpy

## Author
Developed by G. Banidhar
