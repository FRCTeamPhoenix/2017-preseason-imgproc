import cv2
import time
import numpy as np
import platform
import math
import sys

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

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
    focus_absolute = v4l2ctl.get(index, v4l2ctl.PROP_FOCUS_ABSOLUTE)

# set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# intrinsic camera matrix
#
# http://dsp.stackexchange.com/questions/2736/step-by-step-camera-pose-estimation-for-visual-tracking-and-planar-markers
# http://ksimek.github.io/2013/08/13/intrinsic/
#
# [[ au   s   u0  ],
# [  0    av  v0 ],
# [  0    0   1  ]]


#mtx = [[15111.11, 0, 740],
#       [0, 11412.59, 360],
#       [0, 0, 1]]

mtx = [[ 771.82954339,    0,          640.18100339],
       [   0,          777.63905203,  393.76546961],
       [   0,            0,            1        ]]
mtx = np.asarray(mtx)

dist = [[ 0.03236637, -0.03763916, -0.00569912, -0.00091719, -0.008543  ]]
dist = np.asarray(dist)

if cap.isOpened():
    rval, frame = cap.read()
else:
    rval = False

frametimes = list()
last = time.time()

img = cv2.imread("samples/board.png")

while rval:
    # read the framez
    #rval, frame = cap.read()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)

    #epsilon = 0.01 * cv2.arcLength(contours[0], True)
    #polyp = cv2.approxPolyDP(contours[0], epsilon, True)
    #imgp = polyp.astype(np.float32)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        cv2.waitKey(10000);
    sys.exit(0)

    #print()
    #print(imgp)
    #sys.exit(0)

    #print(len(corners2))

    ret, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)


    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(frame, corners2, imgpts)
    cv2.imshow('img', img)
    k = cv2.waitKey(10000) & 0xff
    if k == 's':
        cv2.destroyAllWindows()

    print(rvecs)
    print(tvecs)

    sys.exit(0)

    #with np.load('calib.npz') as X:
    #    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

    #print(mtx)
    #print(approx)


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
