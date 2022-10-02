#importing libraries
import config
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from flask import Flask
from flask import render_template,Response

from flask import Flask,redirect,url_for
from flask_restful import Resource,Api,reqparse
import pandas
#defining path to upload images and to train the model
path = '/Users/aryan/PycharmProjects/FaceRecognition/FaceTrainImages'
images = [] #creating a list to store the images 
classNames = [] #to store the names of images
myList = os.listdir(path)
print(myList)
for each in myList:
    curImage = cv2.imread(f'{path}/{each}')
    images.append(curImage)
    classNames.append(os.path.splitext(each)[0])
print(classNames)

# to find encodings to each one of the image
# creating a function
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# to mark attendance
def markAttendance(name):
    with open('/Users/aryan/PycharmProjects/FaceRecognition/Attendance.csv', 'r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  nameList:
            now = datetime.now()
            timeString = now.strftime('%H:%M:%S')
            daString = now.strftime('%m/%d/%Y')
            f.writelines(f'\n{name},{timeString},{daString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

app=Flask(__name__)
api = Api(app)
camera = cv2.VideoCapture(0)
def gen_frames():  # generate frame by frame from camera
    while True:
        success, img = camera.read()
        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # iterates all the faces that we have found in our current frame and compare
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
            else :
                name = "Unknown"

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
