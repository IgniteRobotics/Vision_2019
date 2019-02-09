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
# these are "square" to the floor.  i.e. the long edge of the tape is vertical
# this is a lie, but may be useful to calculate if the tape is right or left side.
obj_points = np.array([[0, 0, 0], [2, 0, 0], [2, 5.75, 0], [0, 5.75, 0]], np.float32)

# paths to the cameraMatrix and distortMatrix files
cameraMatrix_filepath = "C:/Users/Miriam/Documents/Computer Vision/Cam Calibrate/cameraMatrix.pkl"
distortMatrix_filepath = "C:/Users/Miriam/Documents/Computer Vision/Cam Calibrate/distortMatrix.pkl"

cameraMatrix = pickle.load(open(cameraMatrix_filepath, "rb")) 
distortMatrix = pickle.load(open(distortMatrix_filepath, "rb"))

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

def contour_comparator(a, b):
	return len(a) > len(b)

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

axis = np.float32([[6,0,0], [0,6,0], [0,0,6]]).reshape(-1,3)						   

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)
    return img

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())

frame = None

if args.get("image", False):
	frame = cv2.imread(args.get("image"))
else:
	print("no file to read!")
	exit(-1)

# allow the camera or video file to warm up
time.sleep(2.0)

cv2.namedWindow("frame", 0)
cv2.resizeWindow("frame", 640,480)


size = frame.shape

focal_length = size[1]
cam_center = (size[1]/2, size[0]/2)
zero_camera_matrix = np.array(
                         [[focal_length, 0, cam_center[0]],
                         [0, focal_length, cam_center[1]],
                         [0, 0, 1]], dtype=np.float32
                         )


zero_distort_matrix = np.zeros((4,1))

rvec = None
tvec = None
euler_angle = None
final_pose = None
reprojectdst = None

frame_height, frame_width = frame.shape[:2] 

corners = [np.array([0, 0]), np.array([frame_width, 0]), np.array([frame_width, frame_height]), np.array([0, frame_height])]

mid_frame = (int(frame_width / 2), int(frame_height / 2))

frame_hsv = rid_noise(frame)


# find contours in thresholded frame, then grab the largest one
cnts = cv2.findContours(frame_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("count of contours:", len(cnts))
cnts = imutils.grab_contours(cnts)

c = None

# Get the biggest contour
c = cnts[0]
for contour in cnts:
	if contour_comparator(contour, c):
		c = contour

# print(c)
#
if c is not None and len(c) != 0:
	# cornerPoints = contour_angle(frame, c)

	M = cv2.moments(c)
	center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	cv2.circle(frame, center, 5, (0, 0, 255), -1)
	cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

	# find the corners of the contour
	cornerPoints = get_corners(corners, c)


	# Top left (Blue)
	cv2.circle(frame, (cornerPoints[0][0], cornerPoints[0][1]), 5, (255, 0, 0))
	# Top right (Red)
	cv2.circle(frame, (cornerPoints[1][0], cornerPoints[1][1]), 5, (0, 0, 255))
	# Bottom right (white)
	cv2.circle(frame, (cornerPoints[2][0], cornerPoints[2][1]), 5, (255, 255, 255))
	# Bottom left (yellow)
	cv2.circle(frame, (cornerPoints[3][0], cornerPoints[3][1]), 5, (0, 255, 255))

	#initialize some arrays
	rot = np.array([0])
	
	#_, rvec, tvec = cv2.solvePnP(obj_points, np.array(cornerPoints), cameraMatrix, distortMatrix)
	_, rvec, tvec = cv2.solvePnP(obj_points, np.array(cornerPoints), zero_camera_matrix, zero_distort_matrix)
	
	print('===== tvec =====')
	print(tvec)

	rot, jac = cv2.Rodrigues(rvec)

	#axispts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix,distortMatrix)
	axispts, jac = cv2.projectPoints(axis, rvec, tvec, zero_camera_matrix,zero_distort_matrix)

	#get the angle to the contour
	t = (math.asin(-rot[0][2]))
	#get individual positions.
	x = tvec[0][0]
	y = tvec[1][0]
	z = tvec[2][0]

	distance = math.sqrt(x**2 + z**2)

	# find angles
	Rx = y * (math.cos((math.pi/2) - t))
	Ry = y * (math.sin((math.pi/2) - t))
	ang_bot_to_target = math.atan2(x, z)

	#inverse of rot mat
	rot_inv = rot.transpose()
	p0_world = np.matmul(-rot_inv, tvec)
	angle_target_to_bot = math.atan2(p0_world[0][0], p0_world[2][0])

	print("x", x, "y", y, "z", z)
	print("angle from bot to target", ang_bot_to_target)
	print("angle from target to bot", angle_target_to_bot)
	print("distance", distance)

	cv2.putText(frame, "Distance: " + "{:7.2f}".format(distance), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)
	cv2.putText(frame, "Heading: " + "{:7.2f}".format(ang_bot_to_target), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)

	#add the axis
	frame = draw(frame,cornerPoints,axispts)


cv2.imshow("frame", frame)
cv2.imwrite("out.jpg", frame)
key = cv2.waitKey(-1) & 0xFF

if key == ord("q"):
	exit(1)
	cv2.destroyAllWindows()
	print("okay")