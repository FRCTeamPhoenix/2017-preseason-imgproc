# TO DO
#   - Fix objpoints
#   - Create imgpoints array
#   - Learn more about GPU optimization


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

def draw_poly(img, polyp):
    l = len(polyp)
    for i in range(0, l):
        if i + 1 == l:
            img = cv2.line(img, tuple(polyp[i].ravel()), tuple(polyp[0].ravel()), (255, 128, 0), 5);
        else:
            img = cv2.line(img, tuple(polyp[i].ravel()), tuple(polyp[i + 1].ravel()), (255, 128, 0), 5);


def config_linux(index):
    import v4l2ctl
    #v4l2ctl.restore_defaults(index)
    v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_AUTO, 1)
    v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_AUTO_PRIORITY, 0)
    v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_ABS, 10)
    v4l2ctl.set(index, v4l2ctl.PROP_WHITE_BALANCE_TEMP_AUTO, 0)
    v4l2ctl.set(index, v4l2ctl.PROP_FOCUS_AUTO, 0)

# camera init
index = 0
cap = cv2.VideoCapture("output.avi")

# set the resolution
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
        print("Found camera")
        # run linux specific config using v4l2 driver if the platform is linux
        if platform.system() == "Linux":
            pass
            #config_linux(index)
else:
    rval = False
    print("Did not find camera")

# vars for calculating fps
frametimes = list()
last = time.time()


# loop for as long as we're still getting images
while rval:
    # read the frame
    rval, frame = cap.read()

    # convert to hsv colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([70, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # erode, then dilate to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # get a list of continuous lines in the image
    _, contours, _ = cv2.findContours(mask, 1, 2)

    # there's probably only a target if there are lines in the image
    if (len(contours) > 0):

        # generate object points array
        #objp_outer = np.zeros((4, 3), np.float64)
        #objp_outer[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)
        objp = np.array([[0 ,0 ,0],
                         [1, 0, 0],
                         [0, 1, 0],
                         [1, 1, 0],
                         [1/12, 1/12, 0],
                         [11/12, 1/12, 0],
                         [1/12, 11/12, 0],
                         [11/12, 11/12, 0]])
        #objp = np.append(objp_outer, objp_inner)

        # axis for drawing the debug representation
        axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)

        polygons = []
        for c in contours:
            epsilon = 0.1 * cv2.arcLength(c, True)
            polygon = cv2.approxPolyDP(c, epsilon, True)
            polygons.append((polygon, cv2.contourArea(polygon)))
        polygons = sorted(polygons, key=lambda contour: contour[1])

        #print(objp)
        outer_poly = None
        inner_poly = None
        for p, a in reversed(polygons):
            if len(p) == len(objp) / 2:
                if outer_poly is None:
                    outer_poly = p
                else:
                    inner_poly = p
                    break


        if outer_poly is not None and inner_poly is not None:
            polyp = np.concatenate((outer_poly, inner_poly))
            imgp = polyp.astype(np.float32)

            # calculate rotation and translation matrices
            _, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

            draw_poly(frame, outer_poly)
            draw_poly(frame, inner_poly)

            # show image/check for exit
            #imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            #frame = draw(frame, polyp, imgpts)

    # calculate fps
    frametimes.append(time.time() - last)
    if len(frametimes) > 60:
        frametimes.pop(0)
    fps = int(1 / (sum(frametimes) / len(frametimes)))

    cv2.putText(frame, str(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Debug Display", frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    # record time for fps calculation
    last = time.time()
    #sys.exit(0)

cv2.destroyWindow("Debug Display")
