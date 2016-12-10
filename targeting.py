# TO DO
#   - Implement consistent origin checks (check)
#   - Reprojected points error checking (not worth it, we got rid of that stuff in the cpp proj
#   - Fancy debug features (yeah boy)
#   - Clean code (pretty much)
#   - Only do calculations when the RIO wants them/reset origin when we start aiming
#   - 3D transformation matrix


import cv2
import time
import numpy as np
import platform
from enum import Enum
import math
import sys
import logging
from networktables import NetworkTable

#####################
##### CONSTANTS #####
#####################

# mode
LIVE = True
SHOW_IMAGE = True
DRAW = True
WAIT_FOR_CONTINUE = False
SOURCE = 1
WAIT_TIME = 25
START_FRAME = 1145 # ORIGIN SHIFTS AT THIS TIME, check out ASAP
WINDOW_NAME = "Debug Display"

# controls (ascii)
EXIT = 27
CONTINUE = 32

# logging
STREAM = sys.stdout
LEVEL = logging.DEBUG

# networktables
TABLE_NAME = "jetson_table"
STATE_JETSON = "jetson_state"
class Jetson(Enum):
    POWERED_ON = 1
    CAMERA_ERROR = 2
    TARGET_FOUND = 3
    TARGET_NOT_FOUND = 4

# camera settings
CAM_INDEX = 1
RESOLUTION_X = 1280
RESOLUTION_Y = 720
EXPOSURE_ABS = 10

# image processing settings
THRESH_LOW = np.array([70, 50, 50])
THRESH_HIGH = np.array([80, 255, 255])
MORPH_KERNEL_WIDTH = 3
MORPH_KERNEL_HEIGHT = 3
POLY_EPS = 0.1

# accepted error
MIN_TARGET_AREA = 0.003 * RESOLUTION_X * RESOLUTION_Y
MAX_TARGET_AREA = 0.3 * RESOLUTION_X * RESOLUTION_Y
MIN_NORM_TVECS = 0.0001
MAX_NORM_TVECS = 1000
BAD_ESTIMATE = 8

# drawing settings
LINE_WIDTH = 2
TEXT_SIZE = 1
TEXT_STROKE = 2

# colors
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

# experimentally determined camera (intrinsic) and distortion matrices, converted to numpy arrays
mtx = [[ 771.,      0.,    640.],
       [   0.,    771.,    393.],
       [   0.,      0.,      1.]]
mtx = np.asarray(mtx)
dist = [[ 0.03236637, -0.03763916, -0.00569912, -0.00091719, -0.008543  ]]
dist = np.asarray(dist)

# object points array
objp = np.array([[12, -12,  0],
                 [12,  12,  0],
                 [-12, 12,  0],
                 [-12,-12,  0]], dtype=float)

# axis for drawing the debug representation
axis = np.array([[ 0,  0,  0],
                 [12,  0,  0],
                 [ 0,  12, 0],
                 [ 0,  0, 12]], dtype=float)

last_target = None
T = np.zeros((4, 4), dtype=np.float64)

#####################
####### INIT ########
#####################

# initialize logging
logging.basicConfig(stream=STREAM, level=LEVEL)
log = logging.getLogger(__name__)
log.info("OpenCV %s", cv2.__version__)

# initialize networktables
table = NetworkTable.getTable(TABLE_NAME)
table.putString(STATE_JETSON, Jetson.POWERED_ON)
log.info("Sent powered on message on table %s", TABLE_NAME)

# capture init
cap = cv2.VideoCapture(SOURCE)
log.info("Loaded capture from %s %s", "index" if LIVE else "source file", SOURCE)
if not LIVE:
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    log.info("Set position to frame %s", START_FRAME)

# set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_X)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_Y)
log.info("Set resolution to %sx%s", RESOLUTION_X, RESOLUTION_Y)

# find out if the camera is actually working
if cap.isOpened():
    rval, frame = cap.read()

    # run some configuration if everything is good
    if rval:
        log.info("Read from capture successfully")
        # run config using v4l2 driver if the platform is linux and the feed is live
        if platform.system() == "Linux" and LIVE:
            log.info("Running Linux config using v4l2ctl")
            import v4l2ctl
            v4l2ctl.restore_defaults(SOURCE)
            v4l2ctl.set(SOURCE, v4l2ctl.PROP_EXPOSURE_AUTO, 1)
            v4l2ctl.set(SOURCE, v4l2ctl.PROP_EXPOSURE_AUTO_PRIORITY, 0)
            v4l2ctl.set(SOURCE, v4l2ctl.PROP_EXPOSURE_ABS, EXPOSURE_ABS)
            v4l2ctl.set(SOURCE, v4l2ctl.PROP_WHITE_BALANCE_TEMP_AUTO, 0)
            v4l2ctl.set(SOURCE, v4l2ctl.PROP_FOCUS_AUTO, 0)
    else:
        rval = False
        log.critical("Problem reading from capture")
        table.putString(STATE_JETSON, Jetson.CAMERA_ERROR)


else:
    rval = False
    log.critical("Problem opening capture")
    table.putString(STATE_JETSON, Jetson.CAMERA_ERROR)

# vars for calculating fps
frametimes = list()
last = time.time()


#####################
### FUNCTIONALITY ###
#####################

# draws a 3d axis on an image (calculated from pose estimation)
def draw_axis(img, rvecs, tvecs):
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    corner = tuple(np.array(imgpts[0].ravel(), dtype=int))
    img = cv2.line(img, corner, tuple(np.array(imgpts[1].ravel(), dtype=int)), CYAN, LINE_WIDTH)
    img = cv2.line(img, corner, tuple(np.array(imgpts[2].ravel(), dtype=int)), MAGENTA, LINE_WIDTH)
    img = cv2.line(img, corner, tuple(np.array(imgpts[3].ravel(), dtype=int)), YELLOW, LINE_WIDTH)


# draws a polygon on an image
def draw_poly(img, polyp):
    l = len(polyp)
    for i in range(0, l):
        if i + 1 == l:
            img = cv2.line(img, tuple(polyp[i].ravel()), tuple(polyp[0].ravel()), BLUE, LINE_WIDTH);
        else:
            img = cv2.line(img, tuple(polyp[i].ravel()), tuple(polyp[i + 1].ravel()), BLUE, LINE_WIDTH);


# draw rvecs (the numbers) on an image
def draw_rvecs(img, rvecs):
    cv2.putText(img, "rvecs:", (300, 580), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, GREEN, TEXT_STROKE, cv2.LINE_AA)
    cv2.putText(img, str(rvecs[0]), (300, 620), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, GREEN, TEXT_STROKE, cv2.LINE_AA)
    cv2.putText(img, str(rvecs[1]), (300, 660), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, GREEN, TEXT_STROKE, cv2.LINE_AA)
    cv2.putText(img, str(rvecs[2]), (300, 700), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, GREEN, TEXT_STROKE, cv2.LINE_AA)


# draw tvecs (the numbers) on an image
def draw_tvecs(img, tvecs):
    cv2.putText(img, "tvecs:", (10, 580), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, RED, TEXT_STROKE, cv2.LINE_AA)
    cv2.putText(img, str(tvecs[0]), (10, 620), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, RED, TEXT_STROKE, cv2.LINE_AA)
    cv2.putText(img, str(tvecs[1]), (10, 660), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, RED, TEXT_STROKE, cv2.LINE_AA)
    cv2.putText(img, str(tvecs[2]), (10, 700), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, RED, TEXT_STROKE, cv2.LINE_AA)


# thresholds and edge detects image; returns the result after masking and a list of contours
def process_frame(frame):
    # convert to hsv colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # threshold
    mask = cv2.inRange(hsv, THRESH_LOW, THRESH_HIGH)
    # remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_WIDTH, MORPH_KERNEL_HEIGHT))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # get a list of continuous lines in the image
    _, contours, _ = cv2.findContours(mask, 1, 2)

    return res, contours


# finds the target in a list of contours, returns a matrix with the target polygon
def find_target(contours):
    if len(contours) > 0:
        # find the polygon with the largest area
        best_area = 0
        target = None
        for c in contours:
            e = POLY_EPS * cv2.arcLength(c, True)
            polygon = cv2.approxPolyDP(c, e, True)
            area = cv2.contourArea(polygon)
            if area > best_area:
                best_area = area
                target = polygon

        if target is not None:
            correct_number_of_sides = len(target) == len(objp)
            area_within_range = best_area > MIN_TARGET_AREA and best_area < MAX_TARGET_AREA
            target_within_bounds = True
            for p in target:
                # note, array is double wrapped, that's why accessing x and y values here is weird
                if p[0][0] > RESOLUTION_X - 3 or p[0][0] <= 1 or p[0][1] > RESOLUTION_Y - 3 or p[0][1] <= 1:
                    target_within_bounds = False
                    break

            if correct_number_of_sides and area_within_range and target_within_bounds:
                return target
    return None


# estimates the pose of a target, returns rvecs and tvecs
def estimate_pose(target):
    # fix array dimensions (aka unwrap the double wrapped array)
    new = []
    for r in target:
        new.append([r[0][0], r[0][1]])
    imgp = np.array(new, dtype=np.float64)

    # calculate rotation and translation matrices
    _, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

    if cv2.norm(np.array(tvecs)) < MIN_NORM_TVECS or cv2.norm(np.array(tvecs)) > MAX_NORM_TVECS:
        tvecs = None
    if math.isnan(rvecs[0]):
        rvecs = None
    return rvecs, tvecs


#####################
##### MAIN LOOP #####
#####################

if __name__ == "__main__":
    log.info("Entering main loop")
    # loop for as long as we're still getting images
    while rval:
        # read the frame
        rval, frame = cap.read()
        log.debug("Frame number: %s", cap.get(cv2.CAP_PROP_POS_FRAMES))

        res, contours = process_frame(frame)
        target = find_target(contours)
        if target is not None:
            log.debug("Target found!")
            rvecs, tvecs = estimate_pose(target)
            if rvecs is not None and tvecs is not None:
                log.debug("rvecs:\n%s", rvecs)
                log.debug("tvecs:\n%s", tvecs)

                #R = cv2.Rodrigues(rvecs).transpose()
                #tmat = -R * np.array(tvecs)
                #T[0:3][0:3] = R * 1
                #T[0:3][3:4] = tmat * 1
                #T[3][0] = 0
                #T[3][1] = 0
                #T[3][2] = 0
                #T[3][3] = 1

                log.debug("Transformation:\n%s", T)

                if DRAW:
                    draw_poly(frame, target)
                    draw_rvecs(frame, rvecs)
                    draw_tvecs(frame, tvecs)
                    draw_axis(frame, rvecs, tvecs)
            else:
                log.warn("Pose estimation failed")
            last_target = target
        else:
            log.debug("Target not found")
            last_target = None

        # calculate fps
        frametimes.append(time.time() - last)
        if len(frametimes) > 60:
            frametimes.pop(0)
        fps = int(1 / (sum(frametimes) / len(frametimes)))

        # draw fps
        if DRAW:
            cv2.putText(frame, str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, YELLOW, TEXT_STROKE, cv2.LINE_AA)

        if SHOW_IMAGE:
            scale = 1.48
            frame = cv2.resize(frame, (int(RESOLUTION_X * (scale + 0.02)), int(RESOLUTION_Y * scale)), interpolation=cv2.INTER_CUBIC)
            cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(WAIT_TIME)
        if WAIT_FOR_CONTINUE:
            while key != EXIT and key != CONTINUE:
                key = cv2.waitKey(1)
        if key == EXIT:  # exit on ESC
            break

        # record time for fps calculation
        last = time.time()

    log.info("Main loop exited successfully")
    log.info("FPS at time of exit: %s", fps)
    cv2.destroyWindow(WINDOW_NAME)
