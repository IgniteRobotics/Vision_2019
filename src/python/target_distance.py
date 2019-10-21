from collections import deque
from imutils.video import VideoStream
from imutils import paths
import numpy as np
import argparse
import datetime
import time
import cv2
import math
import imutils
from networktables import NetworkTables
import os
import sys

havedisplay = ('DISPLAY' in os.environ) or os.name =='nt'

if(havedisplay):
	print("display present")
else:
	print("display not present")

# initializes network tables
NetworkTables.initialize(server='10.68.29.2')
nwTables = NetworkTables.getTable('Vision')

# initializes out (to be defined later in loop)
out = None

# time stamp
ts = datetime.datetime.now().timestamp()

# hsv color range for LED/reflective tape
greenLower = (37, 110, 42)#(49,110,12)
greenUpper = (87,255,255) #(86,255,255)
MAX_TURN_ANGLE = 35.2 		# half of the horizonal view of 920 cams

# for filtering
MIN_CONTOUR_AREA = 75
# remember for Y, 0 is the top! image should be 480 pixels tall
MIN_Y_COORDINATE = 125

TOP_MAX_AR = 33 # calculated for the tapes, it should be 25, but allow for some buffer
SINGLE_MIN_AR = 2
SINGLE_MAX_AR = 5

# X_OFFSET = 6.0               # inches to midpoint (default left)
# X_OFFSET_LEFT = 6.0          # inches to midpoint
# X_OFFSET_RIGHT = -4.055      # inches to midpoint 
X_OFFSET = 0.0               # inches to midpoint (default left)
X_OFFSET_LEFT = 0.0          # inches to midpoint
X_OFFSET_RIGHT = -10.055      # inches to midpoint 

Z_OFFSET = -21.0             # inches from camera to bumper
TARGET_AIM_OFFSET = -18.0    # 24.0 #inches in front of target

SLIDER_WINDOW = 30 			# number of frames to average accross
SLIDER_VALUES = 6 			# number of vaules to average accross

# give time to settle??
MAX_DISTANCE_STEP = 100


#######  NEW VALUES FOR FEEDBACK LOOP STEERING ######
#incoming image dimension
image_width = 640
image_height = 480

diagonalFOV = math.radians(78)
horizontalAspect = 16
verticalAspect = 9

horizontalFOV = math.radians(70.42) 
verticalFOV = math.radians(43.3)

H_FOCAL_LENGTH = image_width / (2*math.tan((horizontalFOV/2)))
V_FOCAL_LENGTH = image_height / (2*math.tan((verticalFOV/2)))


# pushes array values onto 2d array window, then slices window to maintain its original size
def window_push(window, values):
    i,j = window.shape
    a = np.vstack((window,values))
    a = a[-i:]
    return a

# column-wise average of a 2d array
def vertical_array_avg(window_array):
    return window_array.mean(axis=0)

	
def rid_noise(img):
	#img = cv2.GaussianBlur(img, (3,3), 0) #(11, 11)?
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#undistortMatrix_img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, greenLower, greenUpper)

	# threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
	thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)
	return thresh


def get_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def filter_contours(contours):
	good = []
	for contour in contours:
		area = cv2.contourArea(contour)
		(X,Y) = get_center(contour)
		if area >= MIN_CONTOUR_AREA and Y >= MIN_Y_COORDINATE:
			good.append(contour)
	return good


def slope(x1, y1, x2, y2):
	return np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    #return (y2-y1)/(x2-x1)

def find_best_contour(cnts, mid_frame):
	sm_Dx = float('inf')
	best_contour = cnts[0]
	for contour in cnts:
		peri = cv2.arcLength(contour, True)
		area = cv2.contourArea(contour)
		print('single contour area:', area)
		foundCenter = get_center(contour)
		Dx = abs(foundCenter[0] - mid_frame) 
		if(sm_Dx > Dx):
			sm_Dx = Dx
			best_contour = contour
	
	#return max(cnts, key=cv2.contourArea) 
	return best_contour


# returns the angle to the target coordinate IN DEGREES
def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return yaw

def target_by_pair(contours, mid_frame, frame):
	print('number of contours for targeting:', len(contours))

	#sort the contours by x coordinate
	contours = sorted(contours, key=lambda x: get_center(x)[0])
	#print('X sorted contours: ', contours)

	cv2.drawContours(frame, contours, 0, (0,255,0), 3)
	
	#now find the center X of each pair
	centers = []

	if len(contours) == 1:
		(cx, cy) = get_center(contours[0])
		centers.append(math.floor(cx))

		cv2.circle(frame, (cx, cy), 5, (255, 255, 255))

		print("one x target found")
	else:
		for i in range(len(contours) -1):
			(cx0,cy0) = get_center(contours[i])
			(cx1,cy1) = get_center(contours[i + 1])
			centers.append(math.floor((cx0 + cx1)/2))

			cv2.circle(frame, (cx0, cy0), 5, (255, 255, 255))
			cv2.circle(frame, (cx1, cy1), 5, (255, 255, 255))

	print('X centers of pairs:', centers)
	
	#now find the closest X to center
	sm_Dx = float('inf')
	x_target = 0
	for x in centers:
		if ((abs(x - mid_frame)) < sm_Dx):
			x_target = x
		#put a dot on each center.
		cv2.circle(frame, (x, 240), 5, (0, 255, 255))
	print ('mid-frame', mid_frame, 'closest X:', x_target)
	#bigger dot on the selected center
	cv2.circle(frame, (x_target, 240), 10, (0, 255, 255))

	#find the angle to this target
	target_angle = calculateYaw(x_target, mid_frame, H_FOCAL_LENGTH)
	print('X angle to target:', target_angle)

	return target_angle


###################### MAIN LOOP ######################




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,	help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	vs = VideoStream(src="/dev/visioncam").start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
average = 0
frame_count=0
rvec = None
tvec = None
window = np.zeros((SLIDER_WINDOW, SLIDER_VALUES), dtype=np.float32)
while True:
	print("#################################################################################################")
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame, then we have reached the end of the video
	if frame is None:
		break

	frame_height, frame_width = frame.shape[:2] # --------------------------------------------- might be nicer way outside of loop

	try:
		if out is None:
			out = cv2.VideoWriter('/media/nvidia/logs/output_' + str(int(ts)) + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
	except:
		out = None

	mid_X_frame = frame_width / 2

	frame_corners = [np.array([0, 0]), np.array([frame_width, 0]), np.array([frame_width, frame_height]), np.array([0, frame_height])]

	size = frame.shape
	print("frame size", size)

	focal_length = size[1]
	cam_center = (size[1]/2, size[0]/2)


	frame_hsv = rid_noise(frame)	

	# find contours in thresholded frame, then grab the largest one
	cnts = cv2.findContours(frame_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts = imutils.grab_contours(cnts)

	# filter contours on size, position etc.
	cnts = filter_contours(cnts)
	

	if cnts is not None and (len(cnts) > 0):
		# prints number of contours found to the monitor 
		print("found contours", len(cnts))
		
		# find the biggest contour in the screen (the closest)
		c = find_best_contour(cnts, mid_X_frame)

		# find target angle for feedback driving
		x_angle = 0
		if (len(cnts) < 6):
			x_angle = target_by_pair(cnts, mid_X_frame, frame)

			

		# draw the values at the top-left corner
		cv2.putText(frame, "X Angle: " + "{:7.2f}".format(x_angle), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)

		print('X_ANGLE', x_angle)

		# print to network tables
		nwTables.putNumber('X_ANGLE', x_angle)
		nwTables.putNumber('TARGET_COUNT', len(cnts))

		if out is not None:
			out.write(frame)

		# show the frame to our screen
		if(havedisplay):
			cv2.imshow("frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the 'q' key is pressed, stop the loop
			if key == ord("q"):
				break
	else:
		print("----- NO TARGET FOUND -----")
		nwTables.putNumber('X_ANGLE', 0)
		nwTables.putNumber('TARGET_COUNT', 0)


# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()
	if out is not None:
		out.release()

# close all windows
if(havedisplay):
	cv2.destroyAllWindows()

print("okay")