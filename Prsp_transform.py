import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


def warper(img, src, dst):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    # Minv = cv2.getPerspectiveTransform(dst, src)  # We're going back before the transformation later.
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


if __name__ == "__main__":

    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Load image
    img = cv2.imread('test_images/test6.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Undistort image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

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

    # Warp image
    warped = warper(undist, src, dst)

    # This is for the visualization to check if two lines are parallel(It means transform is successful)
    cv2.polylines(img, np.int32([src]), True, (0, 0, 255), 2)
    cv2.polylines(warped, np.int32([dst]), True, (0, 0, 255), 2)

    # Save a test image
    # cv2.imwrite('output_images/straight_lines1_BD.jpg', warped)

    # Visualize perspective transform
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Undistorted Image', fontsize=20)
    ax2.imshow(warped)
    ax2.set_title('Warped Image', fontsize=20)
    plt.show()