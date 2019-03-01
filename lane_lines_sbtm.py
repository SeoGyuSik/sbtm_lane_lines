import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import pickle
from Binary_threshold import thresh_binary
from Prsp_transform import warper
import Identify_lines as Id
import Calculating as cc
from collections import deque
from moviepy.editor import VideoFileClip

cap = cv2.VideoCapture(1)


def get_mean_fit(a):
    try:
        s0 = 0
        s1 = 0
        s2 = 0
        for fit in a.recent_fit:
            s0 += fit[0]
            s1 += fit[1]
            s2 += fit[2]
        m0 = s0 / len(a.recent_fit)
        m1 = s1 / len(a.recent_fit)
        m2 = s2 / len(a.recent_fit)

    except:
        print("ZeroDivisionError")
    return np.array([m0, m1, m2])


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # polynomial coefficients for the last fitx values of the last n fits of the line
        # self.recent_fit = deque(maxlen=5)

        # averaged polynomial coefficients
        # self.best_fit = get_mean_fit(self)

        # polynomial coefficients for the most recent fit
        # self.current_fit = np.array(self.recent_fit[len(self.recent_fit) - 1])

        # radius of curvature of the line in some units(meters)
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')


class Setting:
    def __init__(self):
        # Camera matrix
        self.mtx = False

        # Distortion coefficients
        self.dist = False

        # Image size
        self.img_size = (1280, 720)

        # Hard-coding 4 source points and 4 destination points
        self.src = np.float32(
            [[(self.img_size[0] / 2) - 55, self.img_size[1] / 2 + 100],
             [((self.img_size[0] / 6) - 10), self.img_size[1]],
             [(self.img_size[0] * 5 / 6) + 60, self.img_size[1]],
             [(self.img_size[0] / 2 + 55), self.img_size[1] / 2 + 100]])
        self.dst = np.float32(
            [[(self.img_size[0] / 4), 0],
             [(self.img_size[0] / 4), self.img_size[1]],
             [(self.img_size[0] * 3 / 4), self.img_size[1]],
             [(self.img_size[0] * 3 / 4), 0]])


left_line = Line()
right_line = Line()
Set = Setting()


def process_image(image):
    # Undistort image
    undist = cv2.undistort(image, Set.mtx, Set.dist, None, Set.mtx)

    # Threshold image
    binary = thresh_binary(undist)

    # Warp image
    warped = warper(binary, Set.src, Set.dst)

    # if (left_line.detected ==1) & (right_line.detected ==1):
    # leftx, lefty, rightx, righty = Id.search_around_poly(warped)

    # else:
    leftx, lefty, rightx, righty = Id.find_lane_pixels(warped)

    left_fit, right_fit = Id.fit_polynomial(leftx, lefty, rightx, righty)

    newwarp = Id.map_lane(warped, left_fit, right_fit, Set.src, Set.dst)

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    left_cuverad, right_cuverad, diff, dir = cc.measure_curvature_real(left_fit, right_fit)
    cv2.putText(result, "Radius of Curvature = " + str((left_cuverad + right_cuverad)/2) + "(m)", (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv2.putText(result, 'Vehicle is ' + str(diff) + 'm ' + dir + ' of center', (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    return result


if __name__ == "__main__":
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load(open("aaa/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    Set.mtx = mtx
    dist = dist_pickle["dist"]
    Set.dist = dist

    while True:
        ret, frame = cap.read()
        # frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        frame = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
        result_img = process_image(frame)
        cv2.imshow('Cam', result_img)
        if cv2.waitKey(1) == 27:
            break



    # image = cv2.imread('test_images/test1.jpg')
    # res = process_image(image)
    # cv2.imwrite('te2.jpg', res)
    # challenge_output = 'output.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    # clip3 = VideoFileClip('project_video.mp4').subclip(0, 5)
    # clip3 = VideoFileClip('project_video.mp4')
    # challenge_clip = clip3.fl_image(process_image)
    # challenge_clip.write_videofile(challenge_output, audio=False)
