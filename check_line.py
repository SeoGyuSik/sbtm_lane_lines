import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt, pi

vertices = np.array([[0,480],[0,350],[200,200],[300,200],[500,400],[500,480],], np.int32)

def nothing(x):
    pass

# named ites for easy reference
barsWindow = 'Bars'
hl = 'H Low'
hh = 'H High'
sl = 'S Low'
sh = 'S High'
vl = 'V Low'
vh = 'V High'

degree = 0

# set up for video capture on camera 0
cap = cv.VideoCapture(1)

# create window for the slidebars
cv.namedWindow(barsWindow, flags=cv.WINDOW_NORMAL)

# create the sliders
cv.createTrackbar(hl, barsWindow, 0, 255, nothing)
cv.createTrackbar(hh, barsWindow, 0, 255, nothing)
cv.createTrackbar(sl, barsWindow, 0, 255, nothing)
cv.createTrackbar(sh, barsWindow, 0, 255, nothing)
cv.createTrackbar(vl, barsWindow, 0, 255, nothing)

# set initial values for sliders
cv.setTrackbarPos(hl, barsWindow, 0)
cv.setTrackbarPos(hh, barsWindow, 255)
cv.setTrackbarPos(sl, barsWindow, 0)
cv.setTrackbarPos(sh, barsWindow, 255)
cv.setTrackbarPos(vl, barsWindow, 0)


# optional argument for trackbars
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    dot = int(angle / 3.14 * 180)

    if dot < 0:
        dot = dot + 180

    elif dot >= 0:
        pass

    global degree

    degree = dot

    cv.putText(img, str(cntr) + str(dot), cntr, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return angle

def roi(img, vertices):
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv.bitwise_and(img, mask)
    return masked

def get_slope(HSVLOW, HSVHIGH):
    ret, frame = cap.read()
    #frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    frame = cv.resize(frame, dsize=(640, 480), interpolation=cv.INTER_AREA)

    # convert to HSV from BGR
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # apply the range on a mask
    mask = cv.inRange(hsv, HSVLOW, HSVHIGH)
    src = cv.bitwise_and(frame, frame, mask=mask)
    src = roi(src, [vertices])

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # Convert image to binary
    _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 1e3 or 1e5 < area:
            continue
        # Draw each contour only for visualisation purposes
        cv.drawContours(src, contours, i, (0, 0, 255), 2);
        # Find the orientation of each shape
        getOrientation(c, src)

    cv.imshow('Masked', src)
    cv.imshow('Camera', frame)

    global degree
    return degree

def get_hsv():
    ret, frame = cap.read()
    #frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    frame = cv.resize(frame, dsize=(640, 480), interpolation=cv.INTER_AREA)

    # convert to HSV from BGR
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # read trackbar positions for all
    hul = cv.getTrackbarPos(hl, barsWindow)
    huh = cv.getTrackbarPos(hh, barsWindow)
    sal = cv.getTrackbarPos(sl, barsWindow)
    sah = cv.getTrackbarPos(sh, barsWindow)
    val = cv.getTrackbarPos(vl, barsWindow)
    vah = 255

    # make array for final values
    HSVLOW = np.array([hul, sal, val])
    HSVHIGH = np.array([huh, sah, vah])

    # apply the range on a mask
    mask = cv.inRange(hsv, HSVLOW, HSVHIGH)
    src = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('HSV', src)
    cv.imshow('Cam', frame)

    return HSVLOW, HSVHIGH

def close():
    cv.destroyAllWindows()
    return