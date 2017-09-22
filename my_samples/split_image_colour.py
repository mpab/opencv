import cv2
import numpy as np

# local modules
import sys
sys.path.append('../samples/python')
from video import create_capture

cam = create_capture(0)

while(1):

    # Take each frame
    _, frame = cam.read()

    (h, w) = frame.shape[:2]
    #print("w=", w, " h=", h)

    image = cv2.resize(frame, (w//2, h//2))

    b = image.copy()
    b[:,:,1] = 0
    b[:,:,2] = 0

    g = image.copy()
    g[:,:,0] = 0
    g[:,:,2] = 0

    r = image.copy()
    r[:,:,0] = 0
    r[:,:,1] = 0

    #b = frame[:,:,0]

    #b, g, r = cv2.split(image)
    #cv2.imshow('mask', mask)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    cv2.imshow('image', image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()