import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0);

face_cascade = cv.CascadeClassifier("./haarcascades_cuda/haarcascade_frontalface_default.xml");
eyes_cascade = cv.CascadeClassifier("./haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml");
while True:
    ret, frame = cap.read();
    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Detect faces
    #(img,scale,minNeigh)
    #scale is the scale that reduced areach image scale
    #eg. 1.1 is 10%
    #the larger the minNeigh, the higher quality, with lesser detections
    faces = face_cascade.detectMultiScale(grey_frame, 1.05, 5);
    eyes = eyes_cascade.detectMultiScale(grey_frame, 1.05, 5);
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2);
    
    for (x, y, w, h) in eyes:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2);
    
    cv.imshow("Webcam",frame);
    key = cv.waitKey(1);
    if (key == ord("q")):
        break;

cv.destroyAllWindows();
cap.release();
