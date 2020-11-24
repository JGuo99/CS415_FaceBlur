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

    for (x, y, w, h) in face:
        # firstMask = np.zeros(img.shape, dtype = 'uint8')
        cv2.ellipse(
            # firstMask,
            img,
            ((x+w//2), math.floor((y+h)/1.5)),
            (w//2, math.floor(h/1.3)),
            0, 0, 360,
            (50, 150, 250),
            2
        )
        # secondMask = cv2.bitwise_and(img, img, mask = firstMask)
        mask = np.zeros(img.shape, dtype = 'uint8')
        # mask = Image.new("BGR", img.size, color="orange")

        blur = cv2.GaussianBlur(img, (5,5), 0)
        # result = Image.composite(blur, img, mask)

        result = np.where(np.array(mask) > 0, np.array(blur), np.array(img)) 
        # imgBlur = cv2.medianBlur(img, 99)
        # faceBlurred = np.where(mask > 0, imgBlur, img)
        
    return result

if __name__ == '__main__':
    faceCascadePath = 'haarcascade_frontalface_alt.xml'
    img = cv2.imread('test3.png')
    detectedImg = faceDetect(faceCascadePath, img)
    
    cv2.imshow("Detected Face", detectedImg)
    cv2.waitKey(0)
