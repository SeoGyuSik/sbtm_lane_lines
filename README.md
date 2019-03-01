

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2_2]: ./output_images/test1_undist.jpg "Calibrated image"
[straight_lines1_binary_thresholded]: ./output_images/straight_lines1_binary_thresholded.jpg "Binary example"
[straight_lines2_binary_thresholded]: ./output_images/straight_lines2_binary_thresholded.jpg "Binary example"
[test1_binary_thresholded]: ./output_images/test1_binary_thresholded.jpg "Binary example"
[test2_binary_thresholded]: ./output_images/test2_binary_thresholded.jpg "Binary example"
[test3_binary_thresholded]: ./output_images/test3_binary_thresholded.jpg "Binary example"
[test4_binary_thresholded]: ./output_images/test4_binary_thresholded.jpg "Binary example"
[test5_binary_thresholded]: ./output_images/test5_binary_thresholded.jpg "Binary example"
[test6_binary_thresholded]: ./output_images/test6_binary_thresholded.jpg "Binary example"
[image_bird_eye_straight_lines1]: ./output_images/3_transformed/straight_lines1.png "Warped image"
[image_bird_eye_straight_lines2]: ./output_images/3_transformed/straight_lines2.png "Warped image"
[image_bird_eye_test1]: ./output_images/3_transformed/test1.png "Warped image"
[image_bird_eye_test2]: ./output_images/3_transformed/test2.png "Warped image"
[image_bird_eye_test3]: ./output_images/3_transformed/test3.png "Warped image"
[image_bird_eye_test4]: ./output_images/3_transformed/test4.png "Warped image"
[image_bird_eye_test5]: ./output_images/3_transformed/test5.png "Warped image"
[image_bird_eye_test6]: ./output_images/3_transformed/test6.png "Warped image"

[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[lines_identified_test1]: ./output_images/general_sliding_window/lines_Identified_test1.jpg "Fit Visual"
[lines_identified_test2]: ./output_images/general_sliding_window/lines_Identified_test2.jpg "Fit Visual"
[lines_identified_test3]: ./output_images/general_sliding_window/lines_Identified_test3.jpg "Fit Visual"
[lines_identified_test4]: ./output_images/general_sliding_window/lines_Identified_test4.jpg "Fit Visual"
[lines_identified_test5]: ./output_images/general_sliding_window/lines_Identified_test5.jpg "Fit Visual"
[lines_identified_test6]: ./output_images/general_sliding_window/lines_Identified_test6.jpg "Fit Visual"
[map_lane_test1]: ./output_images/Warp_back/map_lane_test1.jpg "Output"
[map_lane_test2]: ./output_images/Warp_back/map_lane_test2.jpg "Output"
[map_lane_test3]: ./output_images/Warp_back/map_lane_test3.jpg "Output"
[map_lane_test4]: ./output_images/Warp_back/map_lane_test4.jpg "Output"
[map_lane_test5]: ./output_images/Warp_back/map_lane_test5.jpg "Output"
[map_lane_test6]: ./output_images/Warp_back/map_lane_test6.jpg "Output"
[video1]: ./project_video.mp4 "Video"



### Find Lane Lines on high way (based on materials of Udacity)


### Camera Calibration

#### 1. Compute the camera matrix and distortion coefficients

The code for this step is contained in lines 7 through 42 of the file called `CameraCalibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion correction

Pipeline is in the `practice.py`

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

The code for this step is contained in lines 34 through 42 of the file called `CameraCalibration.py`.

I started by computing the camera calibration and distortion coefficients using the `cv2.calibrationCamera()` function. I applied this distortion correction to the test image(test_images/test1.jpg) using the `cv2.undistort()` function and obtained this result:

![alt text][image2_2]

#### 2. Create a thresholded binary image using color transform and gradient

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 7 through 33 in `Binary_threshold.py`).  

First, I converted image to HLS color space and get S_channel image. And I got binary image using s_channel threshold (170, 255).

Second, I took the derivative in x using sobel x. And I got binary image by thresholding x gradient.

Finally, I combined two binary images.

Here's an example of my output for this step. 

![alt text][straight_lines1_binary_thresholded]

![alt text][straight_lines2_binary_thresholded]

![alt text][test1_binary_thresholded]

![alt text][test2_binary_thresholded]

![alt text][test3_binary_thresholded]

![alt text][test4_binary_thresholded]

![alt text][test5_binary_thresholded]

![alt text][test6_binary_thresholded]

#### 3. Perform a perspective transform

The code for my perspective transform includes a function called `warper()`, which appears in lines 8 through 15 in the file `Prsp_transform.py`. The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  

I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

The result images are above.

![alt text][image_bird_eye_straight_lines1]

![alt text][image_bird_eye_straight_lines2]

![alt text][image_bird_eye_test1]

![alt text][image_bird_eye_test2]

![alt text][image_bird_eye_test3]

![alt text][image_bird_eye_test4]

![alt text][image_bird_eye_test5]

![alt text][image_bird_eye_test6]

#### Identify lane-line pixels and fit their positions with a polynomial

The code for my identifying lane-line pixels includes a function called `find_lane_pixels()`, which appears in lines 11 through 90 in the file `Identify_lines.py`. The `find_lane_pixels()` function takes as input an warped binary image (`binary_warped`).

First, it take a histogram of the bottom one-third of the image and find the peak of the left and right halves of the histogram using `np.argmax()`.

Second, starting from each peak(left and right) I found non-zero pixels using `sliding window method`.

Finally, `find_lane_pixels()` returns leftx, lefty, rightx, righty. (x-y coordinates of non-zero pixels I found using Sliding window method) 
  
The code for my fitting pixels with a polynomial includes a function called `fit_polynomial()`, which appears in lines 112 through 116 in the file `Identify_lines.py`. The `fit_polynomial()` function takes as input an warped binary image (`binary_warped`).

`fit_polynomial()` gets pixels from `find_lane_pixels()` and fit a second order polynomial to each using `np.polyfit` kinda like this:

(And I used highly targeting search in `search_around_poly` when Sanity check passed in `main_video.py`)  

![alt text][lines_identified_test1]

![alt text][lines_identified_test2]

![alt text][lines_identified_test3]

![alt text][lines_identified_test4]

![alt text][lines_identified_test5]

![alt text][lines_identified_test6]

#### Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature of the lane in lines 6 through 33 in my code in `practice.py`.

After I fitted my pixels obtained using sliding window with 2nd polynomial, I converted this coefficients of the polynomial from pixel space to real world space (i.e. pixels to meters).

And I implemented the calculation of radius of curvature which I took from class notes.

I calculated the position of vehicle with respect to center in lines 16 through 22 in my code in `Identify_lines,py`. I calculated the distance automatically in function `find_lane_pixels()`.

If `dif_ctr = (midpoint - (leftx_base + rightx_base)/2) * mx` has plus sign, the vehicle is on the right side of the center.  And if `dif_ctr = (midpoint - (leftx_base + rightx_base)/2) * mx` < 0, the vehicle is on the left side of the center. 

#### Result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 119 through 143 in my code in `Identify_lines.py` in the function `map_lane()`. I got this code on the project tips. Here is an example of my result on a test image:

![alt text][map_lane_test1]

![alt text][map_lane_test2]

![alt text][map_lane_test3]

![alt text][map_lane_test4]

![alt text][map_lane_test5]

![alt text][map_lane_test6]

---

### Pipeline (video)


Pipeline is in the `practice.py`

Here's a [link to my video result](./output3.mp4)

---


