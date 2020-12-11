import glob
import cv2.cv2 as cv2


# Load cascades
left_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_rightear.xml')
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Test for different ear cascade
both_ears_cascade = cv2.CascadeClassifier('cascades/ear_haarcascade.xml')

# image = cv2.imread('img/test/0003.png')
filenames = glob.glob('img/test/*.png')
images = [cv2.imread(img) for img in filenames]


# Convert to grayscale and equalize histogram
image = images[9]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Start detection


# TESTING
left_ear_test = left_ear_cascade.detectMultiScale(gray, 1.3, 3)
right_ear_test = right_ear_cascade.detectMultiScale(gray, 1.3, 3)
both_ear_test = both_ears_cascade.detectMultiScale(gray, 1.5, 3)

for (x, y, w, h) in left_ear_test:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 4)

for (x, y, w, h) in right_ear_test:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

for (x, y, w, h) in both_ear_test:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)

cv2.imshow('Detector', image)

cv2.waitKey()
cv2.destroyAllWindows()
