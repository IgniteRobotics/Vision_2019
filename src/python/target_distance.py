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

# hsv color range for LED/reflective tape
greenLower = (0,73,22) 
greenUpper = (90,255,78) 

TARGET_AIM_OFFSET = 12.0 #24.0 #inches in front of target

# The (x,y,z) points for the corners of the vision target, in the order top left, top right, bottom right, bottom left
left_obj_points = np.array([[0, 0, 0], [1.945, -0.467, 0], [0.605, -6.058, 0], [-1.34, -5.591, 0]], np.float32)
right_obj_points = np.array([[0, 0, 0], [1.945, 0.467, 0], [3.287, -5.591, 0], [1.34, -6.058, 0]], np.float32)
#obj_points = np.array([[0, 0, 0], [1.945, 0.467, 0], [3.287, -5.591, 0], [1.945, -0.467, 0]], np.float32)
obj_points = left_obj_points

# paths to the cameraMatrix and distortMatrix files
#cameraMatrix_filepath = "/home/nvidia/6829/vision/python/cameraMatrix.pkl"
#distortMatrix_filepath = "/home/nvidia/6829/vision/python/distortMatrix.pkl"

# opening / loading the cameraMatrix and distortMatrix files
#cameraMatrix = pickle.load(open(cameraMatrix_filepath, "rb")) 
#distortMatrix = pickle.load(open(distortMatrix_filepath, "rb"))

def get_corners(corners, contour):
    # We can find the corners of a quadrilateral with sides parallel to the edge of the screen by finding the
    # points on the contour that are closest to each corner of the screen.

    # Initialize the minimum distance to each corner, and which point is at that distance.
    min_dist = [math.inf, math.inf, math.inf, math.inf]
    to_ret = [None, None, None, None]

    # Check distances for every point
    for point in contour:
        for i in range(4):
            # norm is the generalized version of the distance formula.
            dist = np.linalg.norm(point[0] - corners[i])
            if dist < min_dist[i]:
                min_dist[i] = dist
                to_ret[i] = np.array(point[0], dtype=np.float32)
    return to_ret

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

def compute_output_values(rvec, tvec):
	'''Compute the necessary output distance and angles'''
	# The tilt angle only affects the distance and angle1 calcs
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
		b_angle = math.asin((b_side*math.sin(c_angle))/c_side)
	except:
		print("borking")
		a_angle = 0
		b_angle = 0
	return c_side, a_angle, b_angle

def contour_comparator(a, b):
	return len(a) > len(b)

def find_best_contour(cnts, mid_frame):
	sm_Dx = float('inf')
	best_contour = cnts[0]
	for contour in cnts:
		peri = cv2.arcLength(contour, True)
		if len(cv2.approxPolyDP(contour,0.04 * peri, True)) == 4:
			foundCenter = get_center(contour)
			Dx = abs(foundCenter[0] - mid_frame) 
			if(sm_Dx > Dx):
				sm_Dx = Dx
				best_contour = contour
	
	#return max(cnts, key=cv2.contourArea) 
	return best_contour



#axis = np.float32([[6,0,0], [0,6,0], [0,0,6]]).reshape(-1,3)						   

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
    #img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    #img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)
    return img




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
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame, then we have reached the end of the video
	if frame is None:
		break

	frame_height, frame_width = frame.shape[:2] # ---------------------------------------------might be nicer way outside of loop

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
	sd = ShapeDetector()

	if cnts is not None and (len(cnts) > 0):
		# prints number of contours found to the monitor 
		print("found contours", len(cnts))
		
		# find the biggest contour in the screen (the closest)
		c = find_best_contour(cnts, mid_X_frame)

		# acquire corner points of the contour
		cPoints = get_corners(frame_corners, c)

		#determine side
		side = None
		if (cPoints[0][0] > cPoints[3][0]): #top left is to the right, so it's the left tape.
			side = "LEFT"
			obj_points = left_obj_points
		else:
			side = "RIGHT"
			obj_points = right_obj_points

		if c is not None and len(c) != 0:

			shape = sd.detect(c)

			cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

			# draw corner points of largest contour on the frame
			if((sd.detect(c) == "rectangle") or (sd.detect(c) == "square")):
				# Top left point
				frame = cv2.circle(frame, (cPoints[0][0], cPoints[0][1]), 5, (0, 0, 255))
				# Top right point
				frame = cv2.circle(frame, (cPoints[1][0], cPoints[1][1]), 5, (0, 0, 255))
				# Bottom right point
				frame = cv2.circle(frame, (cPoints[2][0], cPoints[2][1]), 5, (0, 0, 255))
				# Bottom left point
				frame = cv2.circle(frame, (cPoints[3][0], cPoints[3][1]), 5, (0, 0, 255))

			try:
				# solvepnp magic
				_, rvec, tvec = cv2.solvePnP(obj_points, np.array(cPoints), zero_camera_matrix, zero_distort_matrix)
				#print('tvec', tvec)
					
			except Exception as e:
				print("no", e)

		# calculate the distance, angle1 (angle from line directly straight in front of camera to the line straight between the camera and target)
		calc_distance, calc_angle1, calc_angle2 = compute_output_values(rvec, tvec)
		angle3 = 180-(abs(calc_angle1 * (180 / math.pi))+abs(calc_angle2 * (180 / math.pi)))
		angle3 = angle3 * (math.pi / 180)

		print('distance', calc_distance, 'angle1', calc_angle1 *(180/math.pi), 'angle2', calc_angle2 *(180/math.pi), 'angle3', angle3 *(180/math.pi))
		print("")
		# find angles and side of triangle set forwards from target 
		calc_c_side, calc_a_angle, calc_b_angle = find_triangle(calc_distance, angle3, TARGET_AIM_OFFSET)

		fixed_angleC = abs(angle3 * (180 / math.pi))
		fixed_angleA = abs(calc_a_angle * (180 / math.pi))
		fixed_angleB = abs(calc_b_angle * (180 / math.pi))
		
		#print("")
		#print("OG triangle", abs(calc_angle1 * (180 / math.pi)),abs(calc_angle2 * (180 / math.pi)),180-(abs(calc_angle1 * (180 / math.pi))+abs(calc_angle2 * (180 / math.pi))))
		#print("")
		#print("sides", calc_c_side, calc_distance, TARGET_AIM_OFFSET)
		#print("")
		#print("angles", fixed_angleA, fixed_angleB, fixed_angleC, fixed_angleA+fixed_angleB+fixed_angleC)

		turn1_angle = (calc_angle1 * (180 / math.pi)) - fixed_angleA #calc_a_angle
		print("turn angle 1", turn1_angle)
		print("go distance", calc_c_side)
		turn2_angle = 180 - (180 - fixed_angleB) # calc_b_angle
		print("turn angle 2", turn2_angle)
		print("go distance", TARGET_AIM_OFFSET)
		print("")

		# finds average distance from the distances calulated (helps remove some error)
		average = average * frame_count
		frame_count += 1
		average += calc_distance
		average = average / frame_count

		print("Distance: " + str(calc_distance))
		print("average distance: " + str(average))

		# print to network tables
		nwTables.putNumber('First Turn', turn1_angle)
		nwTables.putNumber('Distance (side c)', calc_c_side)
		nwTables.putNumber('Second Turn', turn2_angle)
		nwTables.putNumber('Distance(=offset)', TARGET_AIM_OFFSET)

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

# close all windows
if(havedisplay):
	cv2.destroyAllWindows()

print("okay")
