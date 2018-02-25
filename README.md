## Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistorted_scene.png "Road Transformed"
[image3]: ./output_images/binary_color_combined.png "Binary Example"
[image4]: ./output_images/unwarped.png "Warp Example"
[image5]: ./output_images/sliding_window.png "Fit Visual"
[image6]: ./output_images/sliding_window_prev.png "Fit Prev Visual"
[image7]: ./output_images/frame_output.png "Output Visual"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and third code cell of the IPython notebook located in "./project.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function and saved the calibration information as a pickle `calibration_pickle.p`.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied the distortion coefficients to one of the test images using the `cv2.undistort()` function and obtained this result:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color thresholds from HSL and LAB colorspace to generate a binary image (code cells 7-9).  Here's the output for this step on all test images. The function `pipeline()` applies undistortion, unwarp and then computes the binary as well as color images (from the binaries).

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarp()`, which appears in the 5th code cell of the IPython notebook. The cell uses source and destination points to compute the transform matrix (`M`). The `unwarp()` function takes as inputs an image (`img`), as well as the transformation matrix (`M`). Another function called `rectify()` is defined in the same cell which undistorts the image and then performs the warp.

I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(575,464),
                  (711,464), 
                  (1049,682), 
                  (258,682)])
dst = np.float32([(258,0),
                  (1049,0),
                  (1049,720),
                  (258,720)])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:| 
| 575, 464      | 258, 0        | 
| 711, 464      | 1049, 0       |
| 1049, 682     | 1049, 720     |
| 258, 682      | 258, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In code cell 10 the function `sliding_window()` applies a histogram to the lower half of the binary image and uses the peaks to trace the the points upwards. A 2nd order polynomial is then fit over these peak points which give is a lane line.


In code cell 12 the function `fit_prev()` uses the points from the `sliding_window()` for finding lines in the next frame. This reduces the search space for finding the peaks. 

An example of polynomial line overlaid with the search windows on a warped binary frame.

![alt text][image5]

An example of search space when using `fit_prev()` function.

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in code cell 14 using `calculate_curvature_and_distance()` function.
It converts the fitted points to world scale and then fits a poly-line over them. Then using the suggested math we find the radii of curvature of both left and right lanes. 

To find the position of the car with respect to lane center, we consider the lane lines at the bottom of the image. We find the mid point between the lane lines and then find how much it deviates from the horizontal mid-point of the image (assumed to be the car center).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 17 in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced was finding the suitable channels to threshold and find the lane lines (lane lines only). I tried using HSV, Sobel X threshold first. That worked perfectly well on test images but was not up to mark on the project video. I searched for more colorspaces and started analyzing thier outputs on frames where the eralier pipeline failed. I came accross LAB color space which was worked well in finding the yellow lines. Which I combined with HSL for white lines, at this point sobel was not needed anymore.

This implementation is not stable for challenge video and the harder challenge is certainly 'brutal!!'. The sharp changes in the road texture that pass the threshold or the extreme light conditions that prevent the lines from passing the 
threshold are the causes of failure. 

To make the implementation robust, we may need to perform some equalization on the color frames which could make the thresholding work for extreme lighting conditions. I am still not sure how to reject the lines formed by steep texture change on the road, that appear as lanes.