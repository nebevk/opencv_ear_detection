import cv2
import cv2.cv2 as cv2
import numpy as np

# Load cascades
left_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_rightear.xml')
right_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_leftear.xml')
ear_cascade = cv2.CascadeClassifier('cascades/ear_haarcascade.xml')
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Get image
image = cv2.imread('img/0006.png')
# Convert to grayscale and equalizehistogram
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

left_ear = left_ear_cascade.detectMultiScale(gray,  1.1, 5, minSize=(32, 32), flags=cv2.CASCADE_SCALE_IMAGE)
right_ear = right_ear_cascade.detectMultiScale(gray, 1.1, 5)

# Find  ROI
for i in np.arange(1.01, 1.99, 0.01):
    # Try to find the best scaling factor
    face = face_cascade.detectMultiScale(gray, i, 5)
    print(i)
    if len(face)==1:
        for (x, y, w, h) in face:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 3)  # black
        break

# for (x,y,w,h) in left_ear:
#     cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 3) #red
#
# for (x,y,w,h) in right_ear:
#     cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3) #green
#
# for (x,y,w,h) in face:
#     cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,0), 3) #black



cv2.imshow('Detector', image)
cv2.waitKey()
cv2.destroyAllWindows()