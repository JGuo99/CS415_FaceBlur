import numpy as np
import math
import cv2
import sys
from PIL import Image

def faceDetect(path, img):
    faceCascade = cv2.CascadeClassifier(path)
    face = faceCascade.detectMultiScale(
        img,
        scaleFactor = 1.1,
        minNeighbors = 3,
        minSize = (25, 25)
    )

    print("Found {0} face(s)!".format(len(face)))

    imgHeight, imgWidth = img.shape[:2]

    mask = np.zeros(img.shape, dtype = 'uint8')
    for (x, y, w, h) in face:
        cv2.ellipse(
            mask,
            ((x+w//2), math.floor((y+h)/1.25)),
            (w//2, math.floor(h/1.3)),
            0, 0, 360,
            (50, 150, 250),
            -1
        )
        blur = cv2.GaussianBlur(img, (15, 15), 1)
        width, height = (50, 50)
        temp = cv2.resize(blur, (width, height), interpolation=cv2.INTER_LINEAR)
        pixel = cv2.resize(temp, (imgWidth, imgHeight), interpolation=cv2.INTER_NEAREST)
        result = np.where(np.array(mask) > 0, np.array(pixel), np.array(img))        
    return result

if __name__ == '__main__':
    faceCascadePath = 'haarcascade_frontalface_alt.xml'
    img = cv2.imread('Images/test5.png')
    detectedImg = faceDetect(faceCascadePath, img)
    
    cv2.imshow("Detected Face", detectedImg)
    cv2.waitKey(0)