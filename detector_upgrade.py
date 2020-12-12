import glob
import cv2.cv2 as cv2
import numpy as np

# Load cascades
left_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_rightear.xml')
both_ears_cascade = cv2.CascadeClassifier('cascades/ear_haarcascade.xml')
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')


def detectFace(image, gray):
    # Make lists with scale factors ranging from 1.01, to 1.99
    scaleFactor = np.arange(1.01, 1.99, 0.01)
    # Detect ears with 3 different cascades
    for i in scaleFactor:
        # Each iteration use slightly bigger scale factor
        faceList = face_cascade.detectMultiScale(gray, i, 5)
        if len(faceList) == 1:
            for (x, y, w, h) in faceList:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 3)  # black)
                roi_gray = gray[y:y + h, :]
                roi_color = image[y:y + h, :]
                cv2.imshow('ROI', roi_color)
                return roi_color, roi_gray


def detectEars(roi_color, roi_gray):
    # Make lists with scale factors ranging from 1.01, to 1.99
    scaleFactor = np.arange(1.01, 1.99, 0.01)
    # Detect ears with 3 different cascades
    for i in scaleFactor:
        # Each iteration use slightly bigger scale factor
        left_ear = left_ear_cascade.detectMultiScale(roi_gray, i, 5)
        print("k left: ", i)
        right_ear = right_ear_cascade.detectMultiScale(roi_gray, i, 5)
        print("k right: ", i)
        both_ears = both_ears_cascade.detectMultiScale(roi_gray, i, 5)
        # When there is first detection draw bounding box
        if len(left_ear) == 1:
            print("Left Ear scale factor:", i)
            for (xle, yle, wle, hle) in left_ear:
                cv2.rectangle(roi_color, (xle, yle), (xle + wle, yle + hle), (255, 0, 0), 2)  # blue
                earSegment = roi_color[yle:yle + hle, xle:xle + wle]
                return earSegment
            break
        if len(right_ear) == 1:
            print("Right Ear scale factor:", i)
            for (xre, yre, wre, hre) in right_ear:
                cv2.rectangle(roi_color, (xre, yre), (xre + wre, yre + hre), (0, 255, 0), 4)  # green
                earSegment = roi_color[yre:yre + hre, xre:xre + wre]
                return earSegment
            break
        if len(both_ears) == 1:
            print("Both Ears scale factor:", i)
            for (xre, yre, wre, hre) in both_ears:
                cv2.rectangle(roi_color, (xre, yre), (xre + wre, yre + hre), (0, 0, 255), 6)  # blue
                earSegment = roi_color[yre:yre + hre, xre:xre + wre]
                return earSegment
            break


def detection(img):
    # Select specific image
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_color, roi_gray = detectFace(image, gray)
    earSegment = detectEars(roi_color, roi_gray)
    cv2.imwrite('img/output/segmentation/earSegment.jpg', earSegment)
    # Show image
    cv2.imwrite('img/output/detection/earSegment.png', image)
    cv2.imshow('Detector', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return image


# Store images from folder in list
filenames = glob.glob('img/test/*.png')
imgList = [cv2.imread(img) for img in filenames]

# Run detection on all images inside /img
for image in imgList:
    detection(image)
