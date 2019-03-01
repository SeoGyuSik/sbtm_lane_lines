import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import pickle
from Binary_threshold import thresh_binary
from Prsp_transform import warper
# from practice import left_line, right_line


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom one-third of the image
    histogram = np.sum(binary_warped[(binary_warped.shape[0] // 3)*2:, :], axis=0)

    # Find the peak of the first and last thirds of the histogram
    # These will be the starting point for the left and right lines
    # midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:np.int(histogram.shape[0] // 2)])
    rightx_base = np.argmax(histogram[np.int(histogram.shape[0] // 2):]) + np.int(histogram.shape[0] // 2)

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 70
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def search_around_poly(left_fit, right_fit, binary_warped):

    # The width of the margin around the previous polynomial to search
    margin = 70

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] - margin) \
                      & (nonzerox < left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] + margin)).nonzero()[0]
    right_lane_inds = ((nonzerox > right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] - margin) \
                       & (nonzerox < right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] + margin)).nonzero()[0]

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def map_lane(binary_warped, left_fit, right_fit, src, dst):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0]))

    # # Combine the result with the original image
    # result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return newwarp


if __name__ == "__main__":
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    print(mtx)
    dist = dist_pickle["dist"]
    print(dist)
    img_size = (1280, 720)

    # Hard-coding 4 source points and 4 destination points
    src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])

    # Make a list of calibration images
    images = glob.glob('test_images/test*.jpg')

    for idx, fname in enumerate(images):

        # Read image
        img = mpimg.imread(fname)

        # Undistort image
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        # Threshold image
        binary = thresh_binary(undist)

        # Warp image
        warped = warper(binary, src, dst)

        leftx, lefty, rightx, righty = find_lane_pixels(warped)

        left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

        ## Visualization ##
        result = map_lane(warped, left_fit, right_fit, src, dst)
        plt.imshow(result)
        plt.savefig('output_images/a/1_Identified_test' + str(idx + 1) + '.jpg')
        plt.clf()



