import cv2
import time
import numpy as np
import platform
import math
import sys

# used for drawing a 3d axis (calculated from pose estimation)
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def config_linux(index):
    import v4l2ctl
    v4l2ctl.restore_defaults(index)
    # v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_AUTO, 1)
    # v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_ABS, 10)
    # v4l2ctl.set(index, v4l2ctl.PROP_WHITE_BALANCE_TEMP_AUTO, 0)
    # v4l2ctl.set(index, v4l2ctl.PROP_FOCUS_AUTO, 0)

# camera init
index = 0
cap = cv2.VideoCapture(index)



# experimentally determined camera (intrinsic) and distortion matrices, converted to numpy arrays
mtx = [[ 771.82954339,    0,          640.18100339],
       [   0,          777.63905203,  393.76546961],
       [   0,            0,            1        ]]
mtx = np.asarray(mtx)
dist = [[ 0.03236637, -0.03763916, -0.00569912, -0.00091719, -0.008543  ]]
dist = np.asarray(dist)

# find out if the camera is actually working
if cap.isOpened():
    rval, frame = cap.read()

    # run some configuration if everything is good
    if rval:
        # set the resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # run linux specific config using v4l2 driver if the platform is linux
        if platform.system() == "Linux":
            config_linux(index)
else:
    rval = False

# vars for calculating fps
frametimes = list()
last = time.time()

# loop for as long as we're still getting images
while rval:
    # read the frame
    rval, frame = cap.read()

    # convert to hsv colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # split up the channels
    h, s, v = cv2.split(hsv)
    # threshold the value channel
    _, v = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY)

    # erode, then dilate to remove noise
    kernel = np.ones((15, 15), np.uint8)
    v = cv2.morphologyEx(v, cv2.MORPH_OPEN, kernel)

    # get a list of continuous lines in the image
    _, contours, _ = cv2.findContours(v, 1, 2)

    # there's probably only a target if there are lines in the image
    if (len(contours) > 0):

        # generate object points array using fancy linear alg for the shape we're targeting
        objp = np.zeros((4, 3), np.float32)
        objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)

        # axis for drawing the debug representation
        axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)

        # fit a polygon to the contour
        epsilon = 0.01 * cv2.arcLength(contours[0], True)
        polyp = cv2.approxPolyDP(contours[0],epsilon,True)
        imgp = polyp.astype(np.float32)

        # calculate rotation and translation matrices
        _, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)
        print(rvecs)
        print(tvecs)

    # calculate fps
    frametimes.append(time.time() - last)
    if len(frametimes) > 60:
        frametimes.pop(0)
    fps = int(60 / (sum(frametimes) / len(frametimes)))

    # show image/check for exit
    cv2.imshow("Debug Display", v)
    key = cv2.waitKey(10)
    if key == 27:  # exit on ESC
        break
    # record time for fps calculation
    last = time.time()

cv2.destroyWindow("Debug Display")