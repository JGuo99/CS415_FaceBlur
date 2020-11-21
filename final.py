import numpy as np
import cv2
import sys

def faceDetect(path, img):
    faceCascade = cv2.CascadeClassifier(path)
    face = faceCascade.detectMultiScale(
        img,
        scaleFactor = 1.1,
        minNeighbors = 3,
        minSize = (25, 25)
    )
    print("Found {0} face(s)!".format(len(face)))

    for (x, y, w, h) in face:
        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (50, 150, 250),
            2
        )
    return img

if __name__ == '__main__':
    faceCascadePath = 'haarcascade_frontalface_alt.xml'
    img = cv2.imread('test.png')
    detectedImg = faceDetect(faceCascadePath, img)

    cv2.imshow("Detected Face", detectedImg)
    cv2.waitKey(0)
