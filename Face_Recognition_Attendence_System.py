import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize video capture from the default camera
video_capture = cv2.VideoCapture(0)

# Load known faces and their encodings
bunny_image = face_recognition.load_image_file("C:/Users/nsban/OneDrive/Documents/New folder/Bunny.jpg")
bunny_encoding = face_recognition.face_encodings(bunny_image)[0]

rajeev_image = face_recognition.load_image_file("C:/Users/nsban/OneDrive/Documents/New folder/Rajeev.jpg")
rajeev_encoding = face_recognition.face_encodings(rajeev_image)[0]

# Create lists of known face encodings and corresponding names
known_face_encodings = [
    bunny_encoding,
    rajeev_encoding
]

known_face_names = [
    "Bunny",
    "Rajeev"
]

# Create CSV file for attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file_path = current_date + '.csv'

try:
    with open(csv_file_path, 'w+', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Name', 'Time'])  # Write header row
        
        # Initialize variables for face recognition and attendance tracking
        students_present = known_face_names.copy()

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all face locations and encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = []
            for top, right, bottom, left in face_locations:
                # Convert top, right, bottom, left to original size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Extract the face image from the original frame
                face_image = frame[top:bottom, left:right]
                # Convert the face image to RGB (as face_recognition library expects RGB)
                rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                # Compute the encoding for the face
                face_encoding = face_recognition.face_encodings(rgb_face_image)
                if face_encoding:
                    face_encodings.append(face_encoding[0])
                else:
                    print("No face encoding found for the detected face.")

            # Recognize faces in the frame
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                
                face_names.append(name)

                # Update attendance and remove recognized students
                if name in students_present:
                    students_present.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    csv_writer.writerow([name, current_time])

            # Display the recognized faces and attendance status on the frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with the name below the face
                cv2.putText(frame, name + ' Present', (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
