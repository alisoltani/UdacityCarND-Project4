import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.image as mpimg
import glob
import random
import os

# Note! All images are loaded as BGR and not RGBs, and all functions that use that colorspace assume BGR

# Settings
# Checkerboard object points for calibration
nx_checkerboard = 9 # number of inside corners in x
ny_checkerboard = 6 # number of inside corners in y
inside_corners = (nx_checkerboard, ny_checkerboard)

class CameraCalibration()
	def __init__(self):
		self.mtx = None
		self.dist = None
		
	def calibrate(self, path, inside_corners=(9,6)):
		imagefiles = glob.glob(os.path.join(path, '*.jpg')) # All images in the camera calibration path 
		# Initialise image and object point arrays
		objpoints = [] # 3D points in real space
		imgpoints = [] # 2D points in image plane
		
		# Prepare object points (0,0,0), (0,1,0), (9,6,0) - all z's should be zero
		objp = np.zeros((inside_corners[0]*inside_corners[1],3), np.float32)
		objp[:,:2] = np.mgrid[0:inside_corners[0], 0:inside_corners[1]].T.reshape(-1,2) 
		
		for filename in imagefiles:
			image = cv2.imread(filename)
			
			# Convert to grayscale
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			
			# Find the chessboard corners
			ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
			
			# If found, draw corners
			if ret == True:
				# Add object points
				objpoints.append(objp)
				# Add image points
				imgpoints.append(corners)	
				
		
		_, self.mtx, self.dist, -, - = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
		print('Calibration finished')
		
	def undistort(self, img)
		return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
		
class BinaryFilter():
	def __init__(self, sobel_thresh=(20, 100), s_thresh = 160, b_thresh = 50, gb_kernel=(3,7)):
		self.sobel_thresh = sobel_thresh # Threshold for Sobel derivative
		self.s_thresh = s_thresh # Threshold for s channel in HLS transform
		self.b_thresh = b_thresh # Threshold for b channel in LAB transform
		self.gb_kernel = gb_kernel # kernel size for Guassian blurring

	def get_binary_image(self, img):
		blur = cv2.GaussianBlur(img, self.gb_kernel, 0)
		
		# Convert to HLS color space and separate the S channel
		# Note: img is the undistorted image
		hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
		s_channel = hls[:,:,2
		
		# Convert to LAB color space and separate the L channel (frayscale) and B channel (yellow)
		lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
		l_channel = lab[:,:,0]
		b_channel = lab[:,:,2]
		
		# In order to cleanup the sobel derivative, we can set all points in the l_channel that are lower than the mean
		# to a fixed value so the derivative is zero
		l_mean = np.int(np.mean(l_channel))
		l_channel[l_channel<l_mean] = l_mean
		
		# Take sobel in x 
		sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
		abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
		scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))	
		# Threshold x gradient
		sxbinary = np.zeros_like(scaled_sobel)
		sxbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

		# Threshold s color channel
		s_thresh_min = s_thresh
		s_thresh_max = 255
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
		
		# Threshold b color channel
		b_thresh_min = b_thresh
		b_thresh_max = 255
		b_binary = np.zeros_like(b_channel)
		b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1
		
		# Combine the binary thresholds
		combined_binary = np.zeros_like(sxbinary)
		combined_binary[(s_binary == 1) | (sxbinary == 1) | (b_binary == 1)] = 1
		
		return combined_binary

class PerspectiveTransform():
	def __init__(self, src=None, dst=None)
		self.src = np.float32(
				[[572, 455],
				 [0, 720],
				 [1280, 720],
				 [700, 455]])
				 
		self.dst = np.float32(
				[[160, 0],
				 [160, 720],
				 [1120, 720],
				 [1120, 0]])
				 
		self.M = []
		self.Minv = []
				 
	def get_perspective_transform(self):
		self.M = cv2.getPerspectiveTransform(self.src, self.dst)
		self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
		
	def apply_perspective_transform(self,img, inverse=False):
		imshape = img.shape
		# Warp onto birds-eye-view
		# Previous region-of-interest mask's function is absorbed by the warp
		if inverse:
			M = self.Minv
		else:
			M = self.M
		return cv2.warpPerspective(img, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, nwindows = 9, windowmargin = 100, minpix = 50):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
		#polynomial coefficients for the most recent fit
		self.previous_fit = [np.array([False])] 
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
		# Choose the number of sliding windows
		self.nwindows = nwindows
		# Set the width of the windows +/- margin
		self.windowmargin = windowmargin
		# Set minimum number of pixels found to recenter window
		self.minpix = minpix
		self.x = []
        self.y = []
		
		
	def split_image(self, img, left=True):
		# First function to be called, seperates left and right image to isolate the left and right lanes
		midpoint = np.int(img.shape[0]/2)
		if left:
			self.img = img[:, 0:midpoint]
		else:
			self.img = img[:, midpoint:img.shape[0]]

	def sliding_window_histogram(self):
		img = self.img
	    histogram = np.sum(img[np.int(img.shape[0]/2):,:], axis=0)
		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((img, img, img))*255
		base = np.argmax(histogram)

		# Set height of windows
		window_height = np.int(img.shape[0]/self.nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = img.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		current = base
		# Set the width of the windows +/- margin
		margin = self.windowmargin
		# Set minimum number of pixels found to recenter window
		minpix = self.minpix
		# Create empty lists to receive left and right lane pixel indices
		lane_inds = []

		# Step through the windows one by one
		for window in range(self.nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = img.shape[0] - (window+1)*window_height
			win_y_high = img.shape[0] - window*window_height
			win_x_low = current - margin
			win_x_high = current + margin
			# Draw the windows on the visualization image
			cv2.rectangle(out_img,(win_x_low,win_y_low),(win_x_high,win_y_high),(0,255,0), 2) 
			# Identify the nonzero pixels in x and y within the window
			good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
			# Append these indices to the lists
			lane_inds.append(good_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_inds) > minpix:
				x_current = np.int(np.mean(nonzerox[good_inds]))

		# Concatenate the arrays of indices
		lane_inds = np.concatenate(lane_inds)

		# Extract left and right line pixel positions
		x = nonzerox[lane_inds]
		y = nonzeroy[lane_inds] 


		# Fit a second order polynomial to each
		fit = np.polyfit(y, x, 2)
		
		self.previous_fit = self.current_fit
		self.current_fit = fit
		self.detected = True
		self.allx.append(x)
		self.ally.append(y)
		self.x = x
		self.y = y
		
		return fit, x, y

	def search_known_window(self):
		# Assume you now have a new warped binary image 
		# from the next frame of video (also called "img")
		# It's now much easier to find line pixels!	
		img = self.img
		nonzero = img.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		
		margin = self.windowmargin
		
		fit = self.current_fit
		
		lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin))) 

		# Again, extract line pixel positions
		x = nonzerox[lane_inds]
		y = nonzeroy[lane_inds] 
		# Fit a second order polynomial to each
		fit = np.polyfit(y, x, 2)
		
		self.previous_fit = self.current_fit
		self.current_fit = fit
		self.detected = True
		self.allx.append(x)
		self.ally.append(y)
		self.x = x
		self.y = y 
		
		return fit, x, y
		
	def lanedetection(self):
		# Choose between histogram search or known search
		
	def get_curvature(self):
		# Define y-value where we want radius of curvature
		# I'll choose the maximum y-value, corresponding to the bottom of the image
		y_eval = self.img.shape[0]
		curverad = ((1 + (2*self.current_fit[0]*y_eval + self.current_fit[1])**2)**1.5) / np.absolute(2*self.current_fit[0])

		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 30/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/700 # meters per pixel in x dimension

		# Fit new polynomials to x,y in world space
		fit_cr = np.polyfit(self.y*ym_per_pix, self.x*xm_per_pix, 2)
		# Calculate the new radii of curvature
		curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
		# Now our radius of curvature is in meters
		self.curvature = curverad
		return curverad 
		
		

