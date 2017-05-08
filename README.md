## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

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

[image1]: ./examples/cameraCalbiration.png "Calibration"
[image2]: ./examples/undistortion.png "Undistorted images"
[image3]: ./examples/undistortion1.png "Undistorted images"
[image4]: ./examples/binary.png "Binary Example"
[image5]: ./examples/warp.png "Warp Example"
[image6]: ./examples/histogram.png "Example histogram to find starting point"
[image7]: ./examples/slidingwindow.png "Sliding Window technique"
[image8]: ./examples/correlation.png "Correlation technique"
[image9]: ./examples/knownwindow.png "Known Window technique"
[image10]: ./examples/drawBack.png "Known Window technique"
[video8]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

There are two IPython notebooks that was created during this project, the first (AdvancedLaneDetection.ipynb) contains all parts of the project except for the video streamlining. The second notebook (lanedetection.ipynb) takes code from the first notebook and makes it into an object oriented format with classes, and the video generation. 
I will mainly refer to the first notebook as it was written with more explanations inbetween, and only refer to the second during video generating section. There is also a lanedetection.py file that generates the video as well.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the fourth code cell of the IPython notebook Advancedlanedetection.ipynb (also in lanedetection.py lines 12-48).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the result shown in the fifth cell of Advancedlanedetection.ipynb. 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

For this step, I use the cv2.calibrateCamera function (that takes the image and object point generated in the previous step and apply that to cv2.undistort to generate images like:
![alt text][image2].
![alt text][image3]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 50 through 93 in `lanedetection.py`).  I use the Sobel transform (in the x dimension), the r channel in the rgb and the s channel in hls. Here's an example of my output for this step.  

![alt text][image4]
First image is from the Sobel transform, second from the s channel, third from r channel, and last is the combined.

More examples are available in the Advancedlanedetection.ipynb notebook.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_perspective_transform()`, which appears in lines 95 through 126 in the file `lanedetection.py` and in the AdvancedLaneDetection.ipynb.  This function generates the (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32(
        [[572, 455],
         [0, 720],
         [1280, 720],
         [700, 455]])

    dst = np.float32(
        [[160, 0],
         [160, 720],
         [1120, 720],
         [1120, 0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Once I undistort, change the image to binary and unwarp the image, the next step is to identify the lane-line pixels. For this step, I splie the image into left and right and generate to Line class objects to track each lane. This object contains the lanedetection() function that is called when the pipeline is started. Originally I had planned on using both sliding histogram technique and using known previous values, but I ended up only using the sliding histogram technique (implemented in lanedetection.py lines 178 to 244). 

In this function, I first split the image and do a histogram on the bottom half, to find an appropriate starting position.
![alt text][image6]

Once that is done, I discard outlier points in the image (which are more than 1.5 standard deviations away from this value), to help reduce false positives. After that I start from the bottom of the image and generate rectangles, and find lane line pixels in those rectangles. I use the mean value of the pixels to update the base value position of the lanes, to be able to track curves. Once all windows are generated, 
I fit my lane lines with a 2nd order polynomial like this

![alt text][image7]

I also tried and implemented the correlation technique, shown in the figure below, but during trials it did not perform as well as the sliding window.

![alt text][image8]

Once the lane-lines are identified, we can use them as starting points for the next frame (this tehcniques result are shown below), but as mentioned before in this project new sliding window wsa used every time.

![alt text][image9]

These images and code used to generate them is available in AdvancedLaneDetection.ipynb

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented this step in lines 338 and 380 through 390 in my code in `lanedetection.py` and also in the AdvancedLaneDetection.ipynb . The curvature is calculated by taking the derivative of the fitted points. The position in the vehicle is calculated in 380 to 390. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

 After joining the left and right images back together, here is an example of my result on a test image:

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One issue that was faced was during shadows and changes in the road texture. This caused the s channel (in the hls transfomr) to cause false positives, which were hard to remedy. I changed the thresholds a bit and it is better, but in the video (around the 41 second mark) a slight error can be noticed. It doesn't seem to be a catastrophic error, but is an issue to focus on further ahead, but due to time restricition in this project I will leave it for a bit later. The lines are bit wobbly in the beginning, but gets better along the way.

I use an IIR filter to introduce some memory in the fitting parameters, so I get some smoothing to the lane detection. The pipleine will likely fail when the road is not flat (hills) due to the hardcoded nature of the warping function, and should be improved for those situations. Also shadows and color changes can cause the lane detection to fail and need to be worked on further.
