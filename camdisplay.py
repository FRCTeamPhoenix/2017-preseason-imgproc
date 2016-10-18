import cv2
import time
import numpy as np
import platform
import math
import sys

index = 1
cap = cv2.VideoCapture(index)

# only run linux specific code on linux
if platform.system() == "Linux":
    import v4l2ctl
    v4l2ctl.restore_defaults(index)
    #v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_AUTO, 1)
    #v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_ABS, 10)
    v4l2ctl.set(index, v4l2ctl.PROP_WHITE_BALANCE_TEMP_AUTO, 0)
    v4l2ctl.set(index, v4l2ctl.PROP_FOCUS_AUTO, 0)

# set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if cap.isOpened():
    rval, frame = cap.read()
else:
    rval = False

frametimes = list()
last = time.time()

#frame = cv2.imread("last.png")

while rval:
    # read the frame
    rval, frame = cap.read()

    # process image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    _, v = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)
    _, contours, _ = cv2.findContours(v, 1, 2)

    # if len(contours) > 0:
    #     for c in contours:
    #         if len(c) >= 4:
    #             p_last = c[0][0]
    #             for p in c[1:]:
    #                 # only one element in the 2d array
    #                 p = p[0]
    #
    #                 angle = math.atan2(p[1] - p_last[1], p[0] - p_last[0])
    #                 print(angle)
    #                 p_last = p

    #epsilon = 0.01 * cv2.arcLength(contours[0], True)
    #approx = cv2.approxPolyDP(contours[0],epsilon,True)



    #print(approx)
    #sys.exit(0)


    # # get most rectangular contour
    # closest = 0
    # closestc = None
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     a1 = cv2.contourArea(c)
    #     a2 = w * h
    #     ratio = min(a1, a2) / max(a1, a2)
    #     if ratio > closest:
    #         closest = ratio
    #         closestc = c
    #
    # # get bounding rect for the most rectangular contour
    # x, y, w, h = cv2.boundingRect(closestc)
    #
    # # draw that rectangle on the display image
    # out = frame
    # #cv2.rectangle(out, (x,y),(x+w,y+h),(0,255,0),2)

    # calculate fps
    frametimes.append(time.time() - last)
    if len(frametimes) > 60:
        frametimes.pop(0)
    fps = int(60 / (sum(frametimes) / len(frametimes)))

    # show image/check for exit
    cv2.imshow("Debug Display", frame)
    key = cv2.waitKey(10)
    if key == 27:  # exit on ESC
        #cv2.imwrite("board.png", frame)
        break
    #record time for fps calculation
    last = time.time()

cv2.destroyWindow("Debug Display")