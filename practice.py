import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import glob
import cv2
import pickle
from Binary_threshold import thresh_binary
from Prsp_transform import warper
import Identify_lines as Id
import Calculating as cc
from collections import deque
from moviepy.editor import VideoFileClip

# cap = cv2.VideoCapture(1)


# Average three coefficients of 2nd polynomial
def get_mean_fit(a):
    s0 = 0
    s1 = 0
    s2 = 0
    for fit in a:
        s0 += fit[0]
        s1 += fit[1]
        s2 += fit[2]
    m0 = s0 / len(a)
    m1 = s1 / len(a)
    m2 = s2 / len(a)

    return np.array([m0, m1, m2], dtype='float')


# Average numbers in array
def get_mean_num(b):
    s = 0
    for num in b:
        s += num
    m = s / len(b)
    return m


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # polynomial coefficients for the last fitx values of the last 5 fits of the line
        self.recent_fit = deque(maxlen=5)

        # averaged polynomial coefficients in last 5 iterations
        self.best_fit = False  # get_mean_fit(line)

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  # np.array(self.recent_fit[max([0, len(self.recent_fit) - 1])])

        # 5 radii of curvature of the line in the last 5 iterations
        # self.recent_radius_of_curvature = deque(maxlen=5)

        # averaged radius of curvature of the line in the last 5 iterations
        self.best_radius_of_curvature = None  # use get_mean_num()

        # radius of curvature of the line in some units(meters)
        # self.current_radius_of_curvature = None  # int

        # 5 distances in meters of vehicle center from the line in last 5 iterations
        self.recent_line_base_pos = deque(maxlen=5)

        # averaged distance in meters of vehicle center from the line in last 5 iterations
        self.best_line_base_pos = None  # use get_mean_num (소수점 아래 둘째까지만 프린트해)

        # current distance in meters of vehicle center from the line
        self.current_line_base_pos = None

        # difference in fit coefficients between last and new fits
        # self.diffs = np.array([0, 0, 0], dtype='float')


# # Define a class to receive the characteristics of each line detection
# class Track:
#     def __init__(self):


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


def parallel_check_setting():
    # Sanity check (Check if two lines are parallel)
    leftx_base = left_line.current_fit[0] * (719 ** 2) + left_line.current_fit[1] * 719 + left_line.current_fit[2]
    rightx_base = right_line.current_fit[0] * (719 ** 2) + right_line.current_fit[1] * 719 + right_line.current_fit[
        2]

    leftx_mid = left_line.current_fit[0] * (360 ** 2) + left_line.current_fit[1] * 360 + left_line.current_fit[2]
    rightx_mid = right_line.current_fit[0] * (360 ** 2) + right_line.current_fit[1] * 360 + right_line.current_fit[
        2]

    leftx_top = left_line.current_fit[0] * (1 ** 2) + left_line.current_fit[1] * 1 + left_line.current_fit[2]
    rightx_top = right_line.current_fit[0] * (1 ** 2) + right_line.current_fit[1] * 1 + right_line.current_fit[
        2]
    dif_base = rightx_base - leftx_base
    dif_mid = rightx_mid - leftx_mid
    dif_top = rightx_top - leftx_top
    return dif_base, dif_mid, dif_top


def show_txt(result):
    left_line.best_radius_of_curvature, right_line.best_radius_of_curvature, left_line.best_line_base_pos \
        , right_line.best_line_base_pos = cc.measure_curvature_real(left_line.best_fit, right_line.best_fit)
    diff = (1280 / 2 - (left_line.best_line_base_pos + right_line.best_line_base_pos) / 2) * 3.7 / 700
    if diff >= 0:
        direction = 'right'
    else:
        direction = 'left'
    cv2.putText(result, "Radius of Curvature = " + str(
        int((left_line.best_radius_of_curvature + right_line.best_radius_of_curvature) / 2)) + "(m)", (30, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    cv2.putText(result, 'Vehicle is ' + "%0.2f" % float(diff) + 'm ' + direction + ' of center', (30, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    return result


def process_image(image):
    # Undistort image
    undist = cv2.undistort(image, Set.mtx, Set.dist, None, Set.mtx)

    # Threshold image
    binary = thresh_binary(undist)

    # Warp image
    warped = warper(binary, Set.src, Set.dst)

    # Check if we need to a whole image using sliding window
    if (left_line.detected == 1) & (right_line.detected == 1):
        leftx, lefty, rightx, righty = Id.search_around_poly(left_line.best_fit, right_line.best_fit, warped)

    else:
        leftx, lefty, rightx, righty = Id.find_lane_pixels(warped)
    try:
        left_line.current_fit, right_line.current_fit = Id.fit_polynomial(leftx, lefty, rightx, righty)
        # Check if each Line() instance has well-detected lines in recent fit (deque whose max len is 5)
        if (len(left_line.recent_fit) == 0) | (len(right_line.recent_fit) == 0):

            # Sanity check (Check if two lines are parallel)
            dif_base, dif_mid, dif_top = parallel_check_setting()
            if (dif_mid > dif_base * 0.8) & (dif_mid < dif_top * 1.2) & (dif_mid > 500) & (dif_mid < 900):
                left_line.recent_fit.append(left_line.current_fit)
                right_line.recent_fit.append(right_line.current_fit)
                left_line.best_fit = get_mean_fit(left_line.recent_fit)
                right_line.best_fit = get_mean_fit(right_line.recent_fit)
                left_line.detected = 1
                right_line.detected = 1

                newwarp = Id.map_lane(warped, left_line.best_fit, right_line.best_fit, Set.src, Set.dst)

                # Combine the result with the original image
                result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
                result = show_txt(result)
                return result
            else:
                cv2.putText(image, "Can not found two parallel lines yet",
                            (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
                return image

        else:
            # Sanity check (Check if two lines are parallel)
            dif_base, dif_mid, dif_top = parallel_check_setting()
            if (dif_mid > dif_base * 0.8) & (dif_mid < dif_top * 1.2) & (dif_mid > 500) & (dif_mid < 900):
                left_line.recent_fit.append(left_line.current_fit)
                right_line.recent_fit.append(right_line.current_fit)
                left_line.best_fit = get_mean_fit(left_line.recent_fit)
                right_line.best_fit = get_mean_fit(right_line.recent_fit)
                left_line.detected = 1
                right_line.detected = 1
                newwarp = Id.map_lane(warped, left_line.best_fit, right_line.best_fit, Set.src, Set.dst)

                # Combine the result with the original image
                result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
                result = show_txt(result)
                return result
            else:
                newwarp = Id.map_lane(warped, left_line.best_fit, right_line.best_fit, Set.src, Set.dst)
                left_line.detected = 0
                right_line.detected = 0
                # Combine the result with the original image
                result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
                result = show_txt(result)
                return result

    except Exception as e:
        print(e)
        newwarp = Id.map_lane(warped, left_line.best_fit, right_line.best_fit, Set.src, Set.dst)

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        result = show_txt(result)
        return result


if __name__ == "__main__":
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    Set.mtx = mtx
    dist = dist_pickle["dist"]
    Set.dist = dist

    # while True:
    #     ret, frame = cap.read()
    #     # frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    #     frame = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
    #     result_img = process_image(frame)
    #     cv2.imshow('Cam', result_img)
    #     if cv2.waitKey(1) == 27:
    #         break

    # image = cv2.imread('test_images/test1.jpg')
    # res = process_image(image)
    # cv2.imwrite('te33.jpg', res)
    # image2 = cv2.imread('test_images/test2.jpg')
    # resu = process_image(image2)
    # cv2.imwrite('te34.jpg', resu)
    challenge_output = 'output_ch_01.mp4'
    # To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    # To do so add .subclip(start_second,end_second) to the end of the line below
    # Where start_second and end_second are integer values representing the start and end of the subclip
    # You may also uncomment the following line for a subclip of the first 5 seconds
    # clip3 = VideoFileClip('project_video.mp4').subclip(0, 5)
    clip3 = VideoFileClip('challenge_video.mp4')
    challenge_clip = clip3.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)
