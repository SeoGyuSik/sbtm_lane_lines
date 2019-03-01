import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((6*9, 3), np.float32)

    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)


    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('xxx/cal_test*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If found, add object points, image points
        print(ret)
        if ret == 1:
            objpoints.append(objp)
            imgpoints.append(corners)


    import pickle
    # Test undistortion on an image
    img = cv2.imread('aaa/cal_test12.jpg')
    print(imgpoints)
    print(objpoints)
    # Do camera calibration given object points and image points
    print(img.shape[1::-1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('aaa/cal_test12_undist.jpg', dst)

    # Save the camera calibration result (Camera matrix(mtx), distortion coefficients(dist)) for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("aaa/wide_dist_pickle.p", "wb"))

    # # To visualize img as RGB which we're used to.
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    #
    # # Visualize undistortion
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.imshow(img)
    # ax1.set_title('Original Image', fontsize=30)
    # ax2.imshow(dst)
    # ax2.set_title('Undistorted Image', fontsize=30)
    # plt.show()
