import cv2.cv2 as cv2
import numpy as np

# Load cascades
left_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_rightear.xml')
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Test for different ear cascade
both_ears_cascade = cv2.CascadeClassifier('cascades/ear_haarcascade.xml')

# Get images
image = cv2.imread('img/0009.png')

# Scale factor list
scaleFactor = np.arange(1.01, 1.99, 0.01)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)


def findScaleFactor(scaleFactorList, detection):
    for factor in scaleFactorList:
        if len(detection) == 1:
            return factor


def detection(image):
    faceList = face_cascade.detectMultiScale(image, findScaleFactor(scaleFactor), 5)
    for (x, y, w, h) in faceList:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 3)  # black
    return image

detection(gray)


cv2.imshow('Detector', image)
cv2.waitKey()
cv2.destroyAllWindows()