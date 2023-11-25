import cv2
import face_recognition
import datetime
import os

# Load the known faces and encodings
known_faces = {}
known_face_encodings = []
for file in os.listdir("known_faces"):
    if file.endswith(".jpg") or file.endswith(".png"):
        image_path = os.path.join("known_faces", file)
        face = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(face)[0]

        # Extract the person's name from the filename
        person_name = os.path.splitext(file)[0]
        known_faces[person_name] = face_encoding
        known_face_encodings.append(face_encoding)

# Initialize the stranger face list and time list
stranger_faces = []
stranger_times = []
# stranger_images = []  # Initialize an empty list to store stranger images
stranger_image_counter = 1

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = face_recognition.face_locations(rgb_frame)

    # Encode the faces in the frame
    face_encodings = face_recognition.face_encodings(rgb_frame, faces)

    # Loop through each detected face
    for (top, right, bottom, left), face_encoding in zip(faces, face_encodings):
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # If the face is a match, get the name of the person
        name = None
        for person_name, known_face_encoding in known_faces.items():
            if True in face_recognition.compare_faces([known_face_encoding], face_encoding):
                name = person_name
                break

        # If the face is not a match, add it to the stranger face list and time list
        if name is None:
            stranger_faces.append(rgb_frame[top:bottom, left:right])
            stranger_times.append(datetime.datetime.now())

            # Extract the stranger face image and save it to the 'stranger_faces' folder
            # stranger_image = rgb_frame[top:bottom, left:right]
            # stranger_images.append(stranger_image)

            # Generate a unique filename for the stranger image
            filename = f"stranger_{stranger_image_counter}.jpg"
            stranger_image_counter += 1
            # Save the stranger image to the 'stranger_faces' folder
            cv2.imwrite(os.path.join("strangerfaces", filename), stranger_faces[-1])

        # Write the name of the person on the frame
        if name is not None:
            cv2.putText(frame, name, (left, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Quit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
with open("stranger_faces.txt", "w") as f:
    for face in stranger_faces:
        f.write(str(face))

with open("stranger_times.txt", "w") as f:
    for time in stranger_times:
        f.write(str(time))

# with open("stranger_images.txt", "w") as f:
#     for img in stranger_images:
#         f.write(str(img))
# os.makedirs("known_faces1", exist_ok=True)
# Save the image file to the folder
# cv2.imwrite(os.path.join("stran_faces", "image.jpg"), stranger_images)
cap.release()
cv2.destroyAllWindows()
