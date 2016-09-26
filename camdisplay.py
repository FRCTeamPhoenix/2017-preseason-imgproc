import cv2
import v4l2ctl
import time
import numpy as np

index = 1

cap = cv2.VideoCapture(index)
v4l2ctl.restore_defaults(index)


v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_AUTO, 1)
v4l2ctl.set(index , v4l2ctl.PROP_EXPOSURE_ABS, 25)

v4l2ctl.set(index, v4l2ctl.PROP_WHITE_BALANCE_TEMP_AUTO, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if cap.isOpened():
    rval, frame = cap.read()
else:
    rval = False

frametimes = list()
last = time.time()
while rval:
    # read the frame
    rval, frame = cap.read()

    # process image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    _, v = cv2.threshold(v, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)
    _, contours, _ = cv2.findContours(v, 1, 2)

    # get most rectangular contour
    closest = 0
    closestc = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        a1 = cv2.contourArea(c)
        a2 = w * h
        ratio = min(a1, a2) / max(a1, a2)
        if ratio > closest:
            closest = ratio
            closestc = c

    # get bounding rect for the most rectangular contour
    x, y, w, h = cv2.boundingRect(closestc)

    # draw that rectangle on the display image
    out = frame
    #cv2.rectangle(out, (x,y),(x+w,y+h),(0,255,0),2)

    # calculate fps
    frametimes.append(time.time() - last)
    if len(frametimes) > 60:
        frametimes.pop(0)
    fps = int(60 / (sum(frametimes) / len(frametimes)))

    # show image/check for exit
    cv2.imshow("Debug Display", out)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    #record time for fps calculation
    last = time.time()

cv2.destroyWindow("Debug Display")