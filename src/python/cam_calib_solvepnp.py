from collections import deque
from imutils.video import VideoStream
from imutils import paths
from shapedetector import ShapeDetector
import numpy as np
import argparse
import time
import cv2
import math
import imutils
import pickle

greenLower = (0,0,250) 
greenUpper = (100,255,255) 

KNOWN_WIDTH_BTWN_TAPE = 11.0 #inches (not exact)

# The (x,y,z) points for the corners of the vision target, in the order top left, top right, bottom right, bottom left
obj_points = np.array([[0, 0, 0], [2, 0, 0], [2, 5.75, 0], [0, 5.75, 0]], np.float32)

# paths to the cameraMatrix and distortMatrix files
cameraMatrix_filepath = "C:/Users/Miriam/Documents/Computer Vision/Cam Calibrate/cameraMatrix.pkl"
distortMatrix_filepath = "C:/Users/Miriam/Documents/Computer Vision/Cam Calibrate/distortMatrix.pkl"
rvecs_filepath = "C:/Users/Miriam/Documents/Computer Vision/Cam Calibrate/rvecs.pkl"
tvecs_filepath = "C:/Users/Miriam/Documents/Computer Vision/Cam Calibrate/tvecs.pkl"

# opening / loading the cameraMatrix and distortMatrix files
cameraMatrix = pickle.load(open(cameraMatrix_filepath, "rb")) 
distortMatrix = pickle.load(open(distortMatrix_filepath, "rb"))
rvec = pickle.load(open(rvecs_filepath, "rb")) 
tvec = pickle.load(open(tvecs_filepath, "rb"))
#ret = pickle.load(open(ret_filepath, "rb")) 

def get_center(contour):
	M = cv2.moments(contour)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	return cX, cY

def get_angle(p1, p2):
	return math.atan2(p1[1] - p2[1], p1[0] - p2[0]) * 180/math.pi

def undistort_img(img, cameraMatrix, distortMatrix):
	h, w = img.shape[:2]
	newcameraMatrix, roi=cv2.getOptimalNewCameraMatrix(cameraMatrix,distortMatrix,(w,h),1,(w,h))
	
	# undistort image
	dst = cv2.undistort(img, cameraMatrix, distortMatrix, None, newcameraMatrix)
	
	# crop the undistored image
	x,y,w,h = roi
	return dst[y:y+h, x:x+w]
	
def rid_noise(img): #, cameraMatrix, distortMatrix):
	# calls method to undistort image
	#undistortMatrix_img = undistort_img(img, cameraMatrix, distortMatrix)

	#blurred = cv2.GaussianBlur(img, (5, 5), 0) #(11, 11)?
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#undistortMatrix_img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, greenLower, greenUpper)

	# threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
	thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)
	return thresh

def contour_angle(img, c):
	# determine the most extreme points along the contour
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBottom = tuple(c[c[:, :, 1].argmax()][0])
	
	# draw the outline of the object, then draw each of the
	# extreme points, where the left-most is red, right-most
	# is green, top-most is blue, and bottom-most is teal
#	cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
#	cv2.circle(img, extLeft, 8, (0, 0, 255), -1)      # color: RED
#	cv2.circle(img, extRight, 8, (0, 255, 0), -1)     # color: GREEN
#	cv2.circle(img, extTop, 8, (255, 0, 0), -1)       # color: BLUE
#	cv2.circle(img, extBottom, 8, (255, 255, 0), -1)  # color: LIGHT BLUE // TEAL
	
	#cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
	
#	cv2.putText(img, str(extLeft), extLeft, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
#	cv2.putText(img, str(extRight), extRight, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
#	cv2.putText(img, str(extTop), extTop, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
#	cv2.putText(img, str(extBottom), extBottom, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
	
	return [extLeft, extRight, extTop, extBottom]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,	help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame, then we have reached the end of the video
	if frame is None:
		break

	frame_height, frame_width = frame.shape[:2] # ---------------------------------------------might be nicer way outside of loop

	mid_frame = (int(frame_width / 2), int(frame_height / 2))

	frame = undistort_img(frame, cameraMatrix, distortMatrix)
	frame_hsv = rid_noise(frame)#, cameraMatrix, distortMatrix)
#	frame = undistort_img(frame, cameraMatrix, distortMatrix)

	# find contours in thresholded frame, then grab the largest one
	cnts = cv2.findContours(frame_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	sd = ShapeDetector()

	c = max(cnts, key=cv2.contourArea)

	if c is not None and len(c) != 0:
		cornerPoints = contour_angle(frame, c)

		if((sd.detect(c) == "rectangle") or (sd.detect(c) == "square")):
			#[extLeft, extRight, extTop, extBottom]

			# Top left (Blue)
			frame = cv2.circle(frame, (cornerPoints[0][0], cornerPoints[0][1]), 5, (255, 0, 0))
			# Top right (Red)
			frame = cv2.circle(frame, (cornerPoints[1][0], cornerPoints[1][1]), 5, (0, 0, 255))
			# Bottom right (white)
			frame = cv2.circle(frame, (cornerPoints[2][0], cornerPoints[2][1]), 5, (255, 255, 255))
			# Bottom left (yellow)
			frame = cv2.circle(frame, (cornerPoints[3][0], cornerPoints[3][1]), 5, (0, 255, 255))

		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF

	if len(cnts) > 0:
		# for ever contour in the mask, use it to compute the minimum enclosing circle and centroid
		centers = []
		for contour in cnts:
			((x, y), radius) = cv2.minEnclosingCircle(contour)	
			M = cv2.moments(contour)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			shape = sd.detect(contour)

			# only proceed if the radius meets a minimum size
			if(radius > 25 and (shape == "rectangle" or shape == "square")):
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, center, 5, (0, 0, 255), -1)

				centers.append(center)
				# multiply the contour (x, y)-coordinates by the resize ratio,
				# then draw the contours and the name of the shape on the image
				cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
				cv2.putText(frame, "shape", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,	0.5, (255, 255, 255), 2)

		if(len(centers) == 2):
			midpoint = (int((centers[0][0] + centers[1][0])/2), int((centers[0][1] + centers[1][1])/2))
			cv2.circle(frame, midpoint, 5, (0, 255, 0), -1)
			cv2.putText(frame, "midpoint", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
			'''
			if(midpoint[0] > mid_frame[0]):
				print("turn right")
			elif(midpoint[0] == mid_frame[0]):
				print("perfect")
			else:
				print("turn left")
			'''
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	#print("running")
	
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()

print("okay")