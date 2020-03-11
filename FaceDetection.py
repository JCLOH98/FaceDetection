import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0);
ret, frame = cap.read();
#screen percent
spercent = 25;

face_cascade = cv.CascadeClassifier("./haarcascades_cuda/haarcascade_frontalface_default.xml");
eyes_cascade = cv.CascadeClassifier("./haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml");

while True:
    ret, frame = cap.read();
    frame_rescale = cv.resize(frame,(int(frame.shape[1]*spercent/100),int(frame.shape[0]*spercent/100)));
    
    #grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grey_frame = cv.cvtColor(frame_rescale, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    #(img,scale,minNeigh)
    #scale is the scale that reduced areach image scale
    #eg. 1.1 is 10%
    #the larger the minNeigh, the higher quality, with lesser detections
    faces = face_cascade.detectMultiScale(grey_frame, 1.05, 5);
    eyes = eyes_cascade.detectMultiScale(grey_frame, 1.05, 5);
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (int(x*100/spercent), int(y*100/spercent)), (int((x+w)*100/spercent), int((y+h)*100/spercent)), (0, 255, 0), 2);
        cv.rectangle(frame_rescale, (x, y), (x+w, y+h), (0, 255, 0), 2);
    
    for (x, y, w, h) in eyes:
        cv.rectangle(frame, (int(x*100/spercent), int(y*100/spercent)), (int((x+w)*100/spercent), int((y+h)*100/spercent)), (255, 0, 0), 2);
        cv.rectangle(frame_rescale, (x, y), (x+w, y+h), (255, 0, 0), 2);
    
    cv.imshow("Webcam",frame);
    #cv.imshow("Webcam",frame_rescale);
    key = cv.waitKey(1);
    if (key == ord("q")):
        break;

cv.destroyAllWindows();
cap.release();
