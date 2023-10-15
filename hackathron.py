# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:01:42 2023

@author: shiva
"""

import os
import pickle
import numpy as np
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerealtime-b91c6-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "facerealtime-b91c6.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap.set(3, 640)
cap.set(4, 480)

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, peopleIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []
faceCurFrame = []
encodeCurFrame = []
face_names = []

process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    
    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
       # Find all the faces and face encodings in the current frame of video
        faceCurFrame = face_recognition.face_locations(small_frame)
        encodeCurFrame = face_recognition.face_encodings(small_frame, faceCurFrame)
        
        face_names = []
        
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            
            if not any(matches):
                name = "Unknown"
            else:
                face_distances = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(face_distances)
                if matches[matchIndex]:
                    id = peopleIds[matchIndex]
                    peopleInfo = db.reference(f'People/{id}').get()
                    if peopleInfo is not None:
                        name = peopleInfo.get('name', 'Unknown')
                    else:
                        name = 'Unknown'
            face_names.append(name)



    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(faceCurFrame, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

cap.release()
cv2.destroyAllWindows()
