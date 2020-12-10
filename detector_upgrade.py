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

# Convert to grayscale and equalize histogram
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Start detection
scaleFactor = np.arange(1.01, 1.99, 0.01)
for i in scaleFactor:
    # Try to find the best scaling factor
    face = face_cascade.detectMultiScale(gray, i, 5)
    if len(face) == 1:
        print("Face scale factor:", i)
        for (x, y, w, h) in face:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 3)  # black
            # Define ear detection region
            roi_gray = gray[y:y+h, :]
            roi_color = image[y:y+h, :]
            # Ear detection
            for k in scaleFactor:
                left_ear = left_ear_cascade.detectMultiScale(roi_gray, k, 5)
                print("k left: ", k)
                right_ear = right_ear_cascade.detectMultiScale(roi_gray, k, 5)
                print("k right: ", k)
                both_ears = both_ears_cascade.detectMultiScale(roi_gray, k, 5)
                if len(left_ear) == 1:
                    print("Left Ear scale factor:", k)
                    for (xle, yle, wle, hle) in left_ear:
                        cv2.rectangle(roi_color, (xle, yle), (xle+wle, yle+hle), (255, 0, 0), 6)  # blue
                        break
                if len(right_ear) == 1:
                    print("Right Ear scale factor:", k)
                    for (xre, yre, wre, hre) in right_ear:
                        cv2.rectangle(roi_color, (xre, yre), (xre+wre, yre+hre), (0, 255, 0), 6)  # green
                        break
                if len(both_ears) == 1:
                    print("Both Ears scale factor:", k)
                    for (xre, yre, wre, hre) in both_ears:
                        cv2.rectangle(roi_color, (xre, yre), (xre + wre, yre + hre), (0, 100, 100), 6)  # brown
                    break
        break

#TESTING
left_ear_test = left_ear_cascade.detectMultiScale(gray, 1.3, 3)
right_ear_test = right_ear_cascade.detectMultiScale(gray, 1.3, 3)

for (x,y,w,h) in left_ear_test:
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,0), 3)

for (x,y,w,h) in right_ear_test:
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,255), 3)


cv2.imshow('Detector', image)
cv2.waitKey()
cv2.destroyAllWindows()