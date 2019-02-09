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

orangeLower = (0, 100, 100)
orangeUpper = (10, 255, 255)

KNOWN_WIDTH_BTWN_TAPE = 11.0 #inches (not exact)

# The (x,y,z) points for the corners of the vision target, in the order top left, top right, bottom right, bottom left
obj_points = np.array([[0, 0, 0], [2, 0, 0], [2, 5.75, 0], [0, 5.75, 0]], np.float32)

# paths to the cameraMatrix and distortMatrix files
cameraMatrix_filepath = "cameraMatrix.pkl"
distortMatrix_filepath = "distortMatrix.pkl"


# opening / loading the cameraMatrix and distortMatrix files
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
	mask = cv2.inRange(hsv, orangeLower, orangeUpper)	

	# threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
	thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)
	return thresh


	
	return [extLeft, extRight, extTop, extBottom]

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


def get_pose(corners):
	# Run solvePnP then rearrange the matrix it gives to get one that's a 3x3 rotation matrix with a 3d position
	# vector concatenated to it on the right, and the vector <0,0,0,1> appended to the bottom.
	_, rvec, tvec = cv.solvePnP(object_corners, np.array(corners), self.cameraMatrix, self.dist_coeffs)
	zyx, _ = cv.Rodrigues(rvec)
	final = np.concatenate((zyx, tvec), axis=1)
	final = np.concatenate((final, self.bottom), axis=0)
	return np.linalg.inv(final)


reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


fs = cv2.FileStorage("cam.yaml", cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    print("Could not open config file!")
    exit(-1)

dist_coeff = fs.getNode("distortion_coefficients").mat()

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

cv2.namedWindow("frame", 0)
cv2.resizeWindow("frame", 640,480)


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

	corners = [np.array([0, 0]), np.array([frame_width, 0]), np.array([frame_width, frame_height]), np.array([0, frame_height])]

	mid_frame = (int(frame_width / 2), int(frame_height / 2))

	frame_hsv = rid_noise(frame)#, cameraMatrix, distortMatrix)
	

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

	if c is not None and len(c) != 0:
		# cornerPoints = contour_angle(frame, c)

		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		cv2.circle(frame, center, 5, (0, 0, 255), -1)

		cornerPoints = get_corners(corners, c)
		print(cornerPoints[0])

		#if((sd.detect(c) == "rectangle") or (sd.detect(c) == "square")):
	

		# Top left (Blue)
		cv2.circle(frame, (cornerPoints[0][0], cornerPoints[0][1]), 5, (255, 0, 0))
		# Top right (Red)
		cv2.circle(frame, (cornerPoints[1][0], cornerPoints[1][1]), 5, (0, 0, 255))
		# Bottom right (white)
		cv2.circle(frame, (cornerPoints[2][0], cornerPoints[2][1]), 5, (255, 255, 255))
		# Bottom left (yellow)
		cv2.circle(frame, (cornerPoints[3][0], cornerPoints[3][1]), 5, (0, 255, 255))


		bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
		zyx = np.array([0])
		final = np.array([0])
        
		_, rvec, tvec = cv2.solvePnP(obj_points, np.array(cornerPoints), cameraMatrix, dist_coeff)

		zyx, _ = cv2.Rodrigues(rvec)

		reprojectdst, _ = cv2.projectPoints(reprojectsrc, rvec, tvec, cameraMatrix,dist_coeff)

		reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))


		pose_mat = cv2.hconcat((zyx, tvec))
		_, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
		final = np.concatenate((zyx, tvec), axis=1)
		final = np.concatenate((final, bottom), axis=0)
		final_pose = np.linalg.inv(final)

		print ('euler ang', euler_angle)
		print("pose", final_pose)

		# for start, end in line_pairs:
		# 	print("ends",reprojectdst[start], reprojectdst[end])
		# 	cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
        
		cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
		cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)
		cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)


	cv2.imshow("frame", frame)
	key = cv2.waitKey(1) & 0xFF

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