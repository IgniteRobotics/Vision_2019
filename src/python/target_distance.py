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
import pickle
from networktables import NetworkTables
from contour_memory import ContourMemory
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
greenLower = (49,110,12) #(56,122,38) #(31,50,30)      # 0,73,22 
greenUpper = (86,255,255) #(90,255,114) #(95,255,255)    # 90,255,78 

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

# The (x,y,z) points for the corners of the vision target, in the order top left, top right, bottom right, bottom left
left_obj_points = np.array([[0, 0, 0], [1.945, -0.467, 0], [0.605, -6.058, 0], [-1.34, -5.591, 0]], np.float32)
right_obj_points = np.array([[0, 0, 0], [1.945, 0.467, 0], [3.287, -5.591, 0], [1.34, -6.058, 0]], np.float32)
top_obj_points = np.array([[0, 0, 0], [11.89, 0, 0], [9.945, -0.467, 0], [1.945, -0.467, 0]], np.float32)

# paths to the cameraMatrix and distortMatrix files
cameraMatrix_filepath = "/home/nvidia/6829/vision/python/cameraMatrix.pkl"
distortMatrix_filepath = "/home/nvidia/6829/vision/python/distortMatrix.pkl"

# opening / loading the cameraMatrix and distortMatrix files
cameraMatrix = pickle.load(open(cameraMatrix_filepath, "rb")) 
distortMatrix = pickle.load(open(distortMatrix_filepath, "rb"))

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

# memory obj for contour tracking
# max_dist_diff : 
# max_area_diff : 
# min_match 	: contour must match a mininum number of frames
# max_misses 	: 
cm = ContourMemory(max_dist_diff=250, max_area_diff=100, min_match=10, max_misses=1)
#picked_cm = ContourMemory(max_dist_diff=250, max_area_diff=100, min_match=4, max_misses=1)

def dist_array(pt_array, pts_array):
	sorted_ls = []
	print("single point array: ", pt_array)
	print("points array: ", pts_array)
	for pt in pts_array:
		sorted_ls.append(np.linalg.norm(pt_array[0] - pt))
	print("sorted list: ", sorted_ls)
	return sorted_ls #2 points in array returned

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist_array(tl[np.newaxis],rightMost)
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def top_order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	(tr, br) = rightMost[np.argsort(rightMost[:, 1]), :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

# pushes array values onto 2d array window, then slices window to maintain its original size
def window_push(window, values):
    i,j = window.shape
    a = np.vstack((window,values))
    a = a[-i:]
    return a

# column-wise average of a 2d array
def vertical_array_avg(window_array):
    return window_array.mean(axis=0)

def undistort_img(img, cameraMatrix, distortMatrix):
	h, w = img.shape[:2]
	newcameraMatrix, roi=cv2.getOptimalNewCameraMatrix(cameraMatrix,distortMatrix,(w,h),1,(w,h))
	
	# undistort image
	dst = cv2.undistort(img, cameraMatrix, distortMatrix, None, newcameraMatrix)
	
	# crop the undistored image
	x,y,w,h = roi
	return dst[y:y+h, x:x+w]
	
def rid_noise(img):
	#img = cv2.GaussianBlur(img, (3,3), 0) #(11, 11)?
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#undistortMatrix_img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, greenLower, greenUpper)

	# threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
	thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)
	return thresh

def compute_output_values(rvec, tvec, X_OFFSET, Z_OFFSET, TARGET_DIST_OFFSET):
	'''Compute the necessary output distance and angles'''
	# adjust tvec for offsets
	tvec[0] += X_OFFSET
	tvec[2] += Z_OFFSET
	tvec[2] += TARGET_DIST_OFFSET
	
	# declaring our x and z based on tvec
	x = tvec[0][0]
	z = tvec[2][0]

	# distance in the horizontal plane between camera and target
	distance = math.sqrt(x**2 + z**2)

	# horizontal angle between camera center line and target
	angle1 = math.atan2(x, z)

	inv_tvec = tvec
	inv_tvec[1][0] = -tvec[1][0]
	inv_tvec[2][0] = -tvec[2][0]

	rot, _ = cv2.Rodrigues(rvec)
	rot_inv = rot.transpose()
	pzero_world = np.matmul(rot_inv, inv_tvec)
	angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])

	angle_t2 = math.atan2(pzero_world[0][0], pzero_world[1][0])
	angle_t3 = math.atan2(pzero_world[1][0], pzero_world[2][0])

	print('posed angles:', angle2*180/math.pi, angle_t2*180/math.pi, angle_t3*180/math.pi)

	# returns distance to target, angle from front of camera to target, and angle from front of target to camera
	# return distance, angle1, angle2

	# change it to return both turns and distances.
	# also fix to degrees instead of radians
	# turn_1, distance_1, turn_2, distance_2
	# the second turn is 180 - angle2
	#turn_2 = 180 - abs(angle2*(180/math.pi))
	# also need to fix it for right vs left turn
	# if angle2 is positive (target turns right to face bot)
	# then turn 2 will be negative (bot turns left to face target)
	# also the target offset is negative, but the distance 2 should be positive
	turn_2 = (angle2*180/math.pi)*(-1)
		
	return angle1*(180/math.pi), distance, turn_2, (TARGET_DIST_OFFSET*-1)

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

def find_triangle(b_side, c_angle, a_side):
	#goal: find c_side, a_angle, b_anglec
	c_side = math.sqrt(a_side**2 + b_side**2 - 2*a_side*b_side*math.cos(c_angle))
	#print('-1 <= x <= 1 : a_angle ', (a_side*math.sin(c_angle))/c_side)
	try: 
		print("working")
		a_angle = math.asin((a_side*math.sin(c_angle))/c_side)
		#print('-1 <= x <= 1 : c_angle ', (b_side*math.sin(c_angle))/c_side)
		b_angle = (math.pi) - (abs(c_angle) + abs(a_angle)) # math.asin((b_side*math.sin(c_angle))/c_side)
	except:
		print("borking")
		a_angle = 0
		b_angle = 0

	return c_side, a_angle, b_angle

def contour_comparator(a, b):
	return len(a) > len(b)

def find_highest_Y_Pts(candidates):
	print('Finding highest Y pts for', len(candidates), 'contours')
	to_ret = []
	try:
		for i in range(2):
			minY = math.inf
			minPt = []
			minYIdx = 0
			for i, element in enumerate(candidates):
				if (element[0][1] < minY):
					minY = element[0][1]	
					minPt = element
					minYIdx = i
			del(candidates[minYIdx])
			to_ret.append(minPt[0])
			to_ret.append(minPt[1])
		print("to return: ", to_ret)
		to_ret = top_order_points(np.array(to_ret))
		return True, np.int0(to_ret) #np.array((to_ret[0],to_ret[2],to_ret[3],to_ret[1]))
	except Exception as e:
		print ('Failed to find points', str(e))
		return False, None

def slope(x1, y1, x2, y2):
	return np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    #return (y2-y1)/(x2-x1)

def pickTapePairs(contours, img):
	candidates = []
	#for cnt in contours:
	#	cv2.drawContours(img, [cnt], 0, (255,255, 255), 3)
	for contour in contours:
		print('==== contour ====')
		perimeter = cv2.arcLength(contour,True)
		epsilon = 0.01*cv2.arcLength(contour,True)
		approx = cv2.approxPolyDP(contour,epsilon,True)
		print ('length:', len(approx))

		rect = cv2.minAreaRect(contour)
		# calculate coordinates of the minimum area rectangle
		box = cv2.boxPoints(rect)
		# normalize coordinates to integers
		box = np.int0(box)
		print('box: ',box)
		cv2.polylines(frame, [box], True, (0,255,255))
		minY = math.inf
		minYIndex = 0
		# find the highest point (minY)
		for i in range(len(box)):
			if box[i][1] < minY:
				minY = box[i][1]
				minYIndex = i
		
		highestPt = (box[minYIndex][0],box[minYIndex][1])

		#find grab the next and prev points for slopes
		prevPtIndex = minYIndex -1
		nextPtIndex = minYIndex +1
		if prevPtIndex == -1:
			prevPtIndex = 3
		if nextPtIndex == 4:
			nextPtIndex = 0

		#determine which one is closest to the Y factor of our point
		nextHighestPt = (box[prevPtIndex][0],box[prevPtIndex][1])
		if box[nextPtIndex][1] < nextHighestPt[1]:
			nextHighestPt = (box[nextPtIndex][0],box[nextPtIndex][1])

		topSlope = slope(highestPt[0],highestPt[1], nextHighestPt[0],nextHighestPt[1])
		
		print('==== contour ====')		
		print('highestPt:', minYIndex, highestPt[0],highestPt[1])
		print('nextHighestPt:', nextHighestPt[0],nextHighestPt[1])
		print('slope:', topSlope)

		#if topSlope > 90:
		#	topSlope = 180 - topSlope
		#if topSlope > 20 or topSlope < 9:  
		#	print('Removing this contour.  Slope', topSlope, 'too tall')
		#	continue

		candidates.append((highestPt, nextHighestPt))
		# draw contours
		print(type(box))
		print(box)
		#cv2.drawContours(img, [box], 0, (0,0, 255), 3)
		#print(contour)
	# print('================')
	
	print("CANIDATES: ", candidates)
	
	success, top_cornerpoints = find_highest_Y_Pts(candidates)
	if success:
		ar = TOP_MAX_AR
		print ('top corner points:', top_cornerpoints)
		min_rect = cv2.minAreaRect(np.array(top_cornerpoints))
		(center, (w,h), theta) = min_rect
		try:
			ar = w / float(h)
		except:
			print("ar failed")

		print ('top of targets aspect ratio:', str(ar))

		cv2.drawContours(img, [top_cornerpoints], 0, (0,255, 0), 1)

		return np.array(top_cornerpoints), img
	else:
		return None, img

def pickFullOrTopCnt(frame, c, corners):
	ar = TOP_MAX_AR
	
	print ('Current single target', c.shape)
	min_rect = cv2.minAreaRect(c)
	(center, (w,h), theta) = min_rect 
	print('single target w/h',w,h)
	try:
		if(w > h):
			ar = w / float(h)
		else:
			ar = h / float(w)
	except:
		print("ar failed")

	print ('Single target aspect ratio:', str(ar))

	# if the single target found is close to a tape (aspect ration 2.75) then use it
	# otherwise, use the top target.
	target_pts = None
	cornerPoints = None
	side = "???"
	if ar > SINGLE_MIN_AR and ar < SINGLE_MAX_AR:
		print ('USING SINGLE TARGET TAPE')
			# find the corners of the contour

		min_rect = cv2.minAreaRect(c)
		#print("C", c)
		box_pts = cv2.boxPoints(min_rect)
		print("BOX POINTS: ", box_pts)
		# normalize coordinates to integers
		min_rect = np.int0(box_pts)
		print("INT: ", min_rect)
		my_box = min_rect.reshape((-1,1,2))
		# draw the tapes we're using
		cv2.polylines(frame, [my_box], True, (0,255,255))

		cornerPoints = order_points(box_pts) #np.array(min_rect))
		print('corner point type', type(cornerPoints))

		# print ('corners: \nshape (as np.array)', np.array(cornerPoints).shape, '\npoints', cornerPoints)
		target_pts = np.array(cornerPoints)

		#determine side
		side = None
		if (target_pts[0][0] > target_pts[3][0]): #top left is to the right, so it's the left tape.
			side = "LEFT"
			obj_points = left_obj_points
			X_OFFSET = X_OFFSET_LEFT
			print ('Using Left Side')
		else:
			side = "RIGHT"
			obj_points = right_obj_points
			X_OFFSET = X_OFFSET_RIGHT
			print ('Using Right Side')
		
		return target_pts, obj_points 
	return None, None
	'''
	else:		
		print ('USING TOPS OF TARGET TAPES')
		cTopPts, frame = pickTapePairs(cnts, frame)
		if cTopPts is not None:
			print ('Current top tape', cTopPts.shape)
			target_pts= np.array(cTopPts)
			#set the points for solvepnp
			obj_points = top_obj_points
			X_OFFSET = X_OFFSET_LEFT
			side = "TOP"
			return target_pts, obj_points
		else:
			return None, None 
	'''
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
	zero_camera_matrix = np.array([[focal_length, 0, cam_center[0]],[0, focal_length, cam_center[1]],[0, 0, 1]], dtype=np.float32)
 
	zero_distort_matrix = np.zeros((4,1))

	#frame = undistort_img(frame, cameraMatrix, distortMatrix) 
	frame_hsv = rid_noise(frame)	

	# find contours in thresholded frame, then grab the largest one
	cnts = cv2.findContours(frame_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts = imutils.grab_contours(cnts)

	# filter contours on size, position etc.
	cnts = filter_contours(cnts)
	
	# memorize contours
	# only do this once per iteration!!
	# cnts = cm.process_contours(cnts)

	if cnts is not None and (len(cnts) > 0):
		# prints number of contours found to the monitor 
		print("found contours", len(cnts))
		
		# find the biggest contour in the screen (the closest)
		c = find_best_contour(cnts, mid_X_frame)

		# find target angle for feedback driving
		x_angle = 0
		if (len(cnts) < 6):
			x_angle = target_by_pair(cnts, mid_X_frame, frame)

		target_pts, obj_points = pickFullOrTopCnt(frame, c, frame_corners)

		if target_pts is not None and len(target_pts) != 0:

			#cv2.drawContours(frame, [target_pts], -1, (0, 255, 0), 2)

			# Top left point (RED)
			frame = cv2.circle(frame, (target_pts[0][0], target_pts[0][1]), 5, (0, 0, 255))
			# Top right point
			frame = cv2.circle(frame, (target_pts[1][0], target_pts[1][1]), 5, (0, 255, 0))
			# Bottom right point
			frame = cv2.circle(frame, (target_pts[2][0], target_pts[2][1]), 5, (255, 0, 0))
			# Bottom left point
			frame = cv2.circle(frame, (target_pts[3][0], target_pts[3][1]), 5, (255, 255, 255))

			try:
				# solvepnp magic
				_, rvec, tvec = cv2.solvePnP(obj_points, target_pts.astype('float32'), cameraMatrix, distortMatrix)
				#_, rvec, tvec = cv2.solvePnP(obj_points, target_pts.astype('float32'), zero_camera_matrix, zero_distort_matrix)
				print("TVEC:  ", tvec)
			except Exception as e:
				print("no", e)
				continue
		else: # nothing found
			print("----- NO TARGET FOUND -----")
			nwTables.putNumber('TURN_1', 0)
			nwTables.putNumber('DISTANCE_1', 0)
			nwTables.putNumber('TURN_2', 0)
			nwTables.putNumber('DISTANCE_2', 0)
			nwTables.putNumber('DIRECT_TURN', 0)
			nwTables.putNumber('DIRECT_DISTANCE', 0)
			continue

		# calculate the distance, angle1 (angle from line directly straight in front of camera to the line straight between the camera and target)
		#calc_distance, calc_angle1, calc_angle2 = compute_output_values(rvec, tvec, X_OFFSET, Z_OFFSET)
		turn_1, distance_1, turn_2, distance_2 = compute_output_values(rvec, tvec.copy(), X_OFFSET, Z_OFFSET, TARGET_AIM_OFFSET)

		# now get the direct turn and distance. pass 0 as the last offset
		turn_direct, distance_direct, _, _ = compute_output_values(rvec, tvec.copy(), X_OFFSET, Z_OFFSET, 0)

		print('turn_1', turn_1, 'distance_1', distance_1, 'turn_2', turn_2, 'distance_2', distance_2, 'direct_turn', turn_direct, 'direct_distance', distance_direct)
		if(turn_1 >= MAX_TURN_ANGLE or turn_direct >= MAX_TURN_ANGLE):
			print("turn angle1 error! too big!!")
		
		[avg_turn1, avg_distance_1, avg_turn2, avg_distance_2, avg_turn_direct, avg_distance_direct] = vertical_array_avg(window)

		# if the distance jumps too much, toss this observation and try again.
		#if(abs(distance_1 - avg_distance_1) > MAX_DISTANCE_STEP and avg_distance_1 != 0):
		#	print("distance changed too much")
		#	print("new distance", distance_1,"avg distance", avg_distance_1)
		#	continue	
			
		window = window_push(window, [turn_1, distance_1, turn_2, distance_2, turn_direct, distance_direct])
		[turn_1, distance_1, turn_2, distance_2, turn_direct, distance_direct] = vertical_array_avg(window)

		# draw the values at the top-left corner
		cv2.putText(frame, "Turn Angle 1: " + "{:7.2f}".format(turn_1), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Go Distance 1: " + "{:7.2f}".format(distance_1), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Turn Angle 2: " + "{:7.2f}".format(turn_2), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Go Distance 2: " + "{:7.2f}".format(distance_2), (20, 110), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Direct Turn: " + "{:7.2f}".format(turn_direct), (20, 140), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Direct Distance: " + "{:7.2f}".format(distance_direct), (20, 170), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "X Angle: " + "{:7.2f}".format(x_angle), (20, 200), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)

		print("turn angle 1", turn_1) 	
		print("go distance", distance_1)
		print("turn angle 2", turn_2)
		print("go distance", distance_2)
		print("turn direct", turn_direct)
		print("distance direct", distance_direct)
		print('X_ANGLE', x_angle)

		# print to network tables
		nwTables.putNumber('TURN_1', turn_1)
		nwTables.putNumber('DISTANCE_1', distance_1)
		nwTables.putNumber('TURN_2', turn_2)
		nwTables.putNumber('DISTANCE_2', distance_2)
		nwTables.putNumber('DIRECT_TURN', turn_direct)
		nwTables.putNumber('DIRECT_DISTANCE', distance_direct)
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
		nwTables.putNumber('TURN_1', 0)
		nwTables.putNumber('DISTANCE_1', 0)
		nwTables.putNumber('TURN_2', 0)
		nwTables.putNumber('DISTANCE_2', 0)
		nwTables.putNumber('DIRECT_TURN', 0)
		nwTables.putNumber('DIRECT_DISTANCE', 0)

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