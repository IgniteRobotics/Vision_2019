from collections import deque
from imutils.video import VideoStream
from imutils import paths
import numpy as np
import argparse
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

# hsv color range for LED/reflective tape
greenLower = (0,73,22) 
greenUpper = (90,255,78) 

MAX_TURN_ANGLE = 35.2 		# half of the horizonal view of 920 cams

X_OFFSET = 6.0              # inches to midpoint (default left)
X_OFFSET_LEFT = 6.0         # inches to midpoint
X_OFFSET_RIGHT = -4.055     # inches to midpoint 
Z_OFFSET = -21.0            # inches from camera to bumper
TARGET_AIM_OFFSET = 24.0    # 24.0 #inches in front of target

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

#memory obj for contour tracking
cm = ContourMemory(max_dist_diff=100, max_area_diff=100, min_match=20, max_misses=1)

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

def compute_output_values(rvec, tvec, X_OFFSET, Z_OFFSET):
	'''Compute the necessary output distance and angles'''
	# adjust tvec for offsets
	tvec[0] += X_OFFSET
	tvec[2] += Z_OFFSET
	
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
	return distance, angle1, angle2

def get_center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

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
		area = cv2.contourArea(contour)
		print ('area:', area)

		if area < 150:
			print('Removing this contour.  Area', area, 'too small')
			continue 

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
		ar = 100
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
	ar = 100
	
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
	if ar > 2 and ar < 4:
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

def find_best_contour(cnts, mid_frame):
	sm_Dx = float('inf')
	best_contour = cnts[0]
	for contour in cnts:
		peri = cv2.arcLength(contour, True)
		area = cv2.contourArea(contour)
		print('single contour area:', area)
		if area < 250:
			continue
		# if len(cv2.approxPolyDP(contour,0.04 * peri, True)) == 4:
		foundCenter = get_center(contour)
		Dx = abs(foundCenter[0] - mid_frame) 
		if(sm_Dx > Dx):
			sm_Dx = Dx
			best_contour = contour
	
	#return max(cnts, key=cv2.contourArea) 
	return best_contour


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
			out = cv2.VideoWriter('/media/nvidia/logs/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
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

	#memorize contours
	#only do this once per iteration!!
	cnts = cm.process_contours(cnts)

	if cnts is not None and (len(cnts) > 0):
		# prints number of contours found to the monitor 
		print("found contours", len(cnts))
		
		# find the biggest contour in the screen (the closest)
		c = find_best_contour(cnts, mid_X_frame)

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
			continue

		# calculate the distance, angle1 (angle from line directly straight in front of camera to the line straight between the camera and target)
		calc_distance, calc_angle1, calc_angle2 = compute_output_values(rvec, tvec, X_OFFSET, Z_OFFSET)

		if(calc_angle1 >= MAX_TURN_ANGLE):
			print("turn angle1 error! too big!!")

		print('distance', calc_distance, 'angle1', calc_angle1 *(180/math.pi), 'angle2', calc_angle2 *(180/math.pi))
		print("")

		#angle to run straight at the target.
		direct_turn = calc_angle1 *(180/math.pi)

		# find angles and side of triangle set forwards from target 
		calc_c_side, calc_a_angle, calc_b_angle = find_triangle(calc_distance, calc_angle2, TARGET_AIM_OFFSET)

		fixed_angleA = calc_a_angle * (180 / math.pi) #removed abs
		fixed_angleB = calc_b_angle * (180 / math.pi) #removed abs
		
		print("fixed_angleA: ", fixed_angleA, "fixed_angleB: ", fixed_angleB)

		turn1_angle = (calc_angle1 * (180 / math.pi)) + fixed_angleA #calc_a_angle
		turn2_angle = 180 - fixed_angleB # calc_b_angle
		
		# if bot is to left of target, turn2 must be a left turn (negative turn)
		if(calc_angle2 > 0):
			turn2_angle = turn2_angle * -1

		[avg_turn1_angle, avg_calc_c_side, avg_turn2_angle, avg_goDist2, avg_calc_distance, avg_direct_turn] = vertical_array_avg(window)
		
		if(abs(calc_distance - avg_calc_distance) > MAX_DISTANCE_STEP and avg_calc_distance != 0):
			print("distance changed too much")
			print("new distance",calc_distance,"avg distance", avg_calc_distance)
			continue	
			
		window = window_push(window, [turn1_angle, calc_c_side, turn2_angle, TARGET_AIM_OFFSET, calc_distance, direct_turn])
		[turn1_angle, calc_c_side, turn2_angle, goDist2, calc_distance, direct_turn] = vertical_array_avg(window)

		# draw the values at the top-left corner
		cv2.putText(frame, "Turn Angle 1: " + "{:7.2f}".format(turn1_angle), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Go Distance 1: " + "{:7.2f}".format(calc_c_side), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Turn Angle 2: " + "{:7.2f}".format(turn2_angle), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Go Distance 2: " + "{:7.2f}".format(goDist2), (20, 110), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Direct Distance: " + "{:7.2f}".format(calc_distance), (20, 140), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)
		cv2.putText(frame, "Direct Turn: " + "{:7.2f}".format(direct_turn), (20, 170), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 255, 255), thickness=2)

		print("turn angle 1", turn1_angle) 	
		print("go distance", calc_c_side)
		print("turn angle 2", turn2_angle)
		print("go distance", goDist2)
		print("direct turn", direct_turn)
		print("direct dist", calc_distance)
		print("")

		# finds average distance from the distances calulated (helps remove some error)
		average = average * frame_count
		frame_count += 1
		average += calc_distance
		average = average / frame_count

		print("Distance: " + str(calc_distance))
		print("average distance: " + str(average))

		# print to network tables
		nwTables.putNumber('TURN_1', turn1_angle)
		nwTables.putNumber('DISTANCE_1', calc_c_side)
		nwTables.putNumber('TURN_2', turn2_angle)
		nwTables.putNumber('DISTANCE_2', goDist2)
		nwTables.putNumber('DIRECT_TURN', direct_turn)
		nwTables.putNumber('DIRECT_DISTANCE', calc_distance)

		if out is not None:
			out.write(frame)

		# show the frame to our screen
		if(havedisplay):
			cv2.imshow("frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the 'q' key is pressed, stop the loop
			if key == ord("q"):
				break

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