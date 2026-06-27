import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
from pathlib import Path

# Initialize video capture from the default camera
video_capture = cv2.VideoCapture(0)

# Load known faces and their encodings
image_path = Path(__file__).resolve().parent / "pp2.jpg"
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

bunny_image = face_recognition.load_image_file(str(image_path))
bunny_encoding = face_recognition.face_encodings(bunny_image)[0]


# Create lists of known face encodings and corresponding names
known_face_encodings = [
    bunny_encoding
]

known_face_names = [
    "Bunny",
]

# Create CSV file for attendance
current_date = datetime.now().strftime("%Y-%m-%d")
csv_file_path = current_date + '.csv'

attendance = {name: {'Time': '', 'Attendence': 'Absent'} for name in known_face_names}
if Path(csv_file_path).exists():
    with open(csv_file_path, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row and row.get('Name') in attendance:
                attendance[row['Name']]['Time'] = row.get('Time', '')
                attendance[row['Name']]['Attendence'] = row.get('Attendence', 'Absent')

# Write attendance dictionary back to CSV so the file has one row per known student
def write_attendance_file():
    with open(csv_file_path, 'w', newline='') as csv_file:
        fieldnames = ['Name', 'Time', 'Attendence']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for name in known_face_names:
            row = {
                'Name': name,
                'Time': attendance[name]['Time'],
                'Attendence': attendance[name]['Attendence']
            }
            csv_writer.writerow(row)

write_attendance_file()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Recognize faces in the frame
        face_names = []
        updated = False
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

            if name in attendance and attendance[name]['Attendence'] != 'Present':
                attendance[name]['Attendence'] = 'Present'
                attendance[name]['Time'] = datetime.now().strftime("%H:%M:%S")
                write_attendance_file()
                updated = True

        # Display the recognized faces and attendance Attendence on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with the name below the face
            label = name + ' Present' if name != 'Unknown' else 'Unknown'
            cv2.putText(frame, label, (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Attendance System', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print("Error occurred while creating or writing to CSV file:", e)

# Release video capture and close CSV file
video_capture.release()
cv2.destroyAllWindows()
