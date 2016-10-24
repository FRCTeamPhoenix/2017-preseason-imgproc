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

while rval:
    # read the frame
    rval, frame = cap.read()

    # show image/check for exit
    cv2.imshow("Debug Display", frame)
    key = cv2.waitKey(10)
    if key == 27:  # exit on ESC
        cv2.imwrite("board.png", frame)
        break
    #record time for fps calculation
    last = time.time()

cv2.destroyWindow("Debug Display")