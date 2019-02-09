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

NetworkTables.initialize(server='10.68.29.2');

nwTables = NetworkTables.getTable('Vision')

greenLower = (0,73,22) 
greenUpper = (90,255,78) 

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

def contour_angle(img, c):
	# determine the most extreme points along the contour
	extLeft = np.array(tuple(c[c[:, :, 0].argmin()][0]), dtype=np.float32)
	extRight = np.array(tuple(c[c[:, :, 0].argmax()][0]), dtype=np.float32)
	extTop = np.array(tuple(c[c[:, :, 1].argmin()][0]), dtype=np.float32)
	extBottom = np.array(tuple(c[c[:, :, 1].argmax()][0]), dtype=np.float32)

#	cv2.putText(img, str(extLeft), extLeft, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
#	cv2.putText(img, str(extRight), extRight, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
#	cv2.putText(img, str(extTop), extTop, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
#	cv2.putText(img, str(extBottom), extBottom, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
	
	return [extLeft, extRight, extTop, extBottom]

def compute_output_values(rvec, tvec): #self, rvec, tvec):
	'''Compute the necessary output distance and angles'''
	# The tilt angle only affects the distance and angle1 calcs
	x = tvec[0][0]
	z = tvec[2][0]

	#z = math.sin(self.tilt_angle) * tvec[1][0] + math.cos(self.tilt_angle) * tvec[2][0]

	# distance in the horizontal plane between camera and target
	distance = math.sqrt(x**2 + z**2)

	# horizontal angle between camera center line and target
	angle1 = math.atan2(x, z)

	rot, _ = cv2.Rodrigues(rvec)
	
	rot_inv = rot.transpose()
	
	pzero_world = np.matmul(rot_inv, -tvec)
	
	angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])
	return distance, angle1, angle2


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

	frame_corners = [np.array([0, 0]), np.array([frame_width, 0]), np.array([frame_width, frame_height]), np.array([0, frame_height])]

	size = frame.shape
	print("frame size", size)

	#cv2.imshow("frame", frame)

	focal_length = size[1]
	cam_center = (size[1]/2, size[0]/2)
	zero_camera_matrix = np.array([[focal_length, 0, cam_center[0]],[0, focal_length, cam_center[1]],[0, 0, 1]], dtype=np.float32)

	zero_distort_matrix = np.zeros((4,1))

	mid_frame = (int(frame_width / 2), int(frame_height / 2))

	#frame = undistort_img(frame, cameraMatrix, distortMatrix) 
	frame_hsv = rid_noise(frame)
	#cv2.imshow("hsv",frame_hsv)
	

	# find contours in thresholded frame, then grab the largest one
	cnts = cv2.findContours(frame_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts = imutils.grab_contours(cnts)
	sd = ShapeDetector()

	if cnts is not None and (len(cnts) > 0):
		print("found contours", len(cnts))
		c = max(cnts, key=cv2.contourArea)
		#_, rvec, tvec = cv2.solvePnP(obj_points, np.array(cornerPoints), cameraMatrix, distortMatrix)
		#reprojectdst, _ = cv2.projectPoints(reprojectsrc, rvec, tvec, cameraMatrix,distortMatrix)
		#cv2.solvePnP()
		
		cPoints = get_corners(frame_corners, c)

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
				#cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
				cv2.drawContours(frame, cnts, -1, (0, 255, 0), 2)
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

		if c is not None and len(c) != 0:
			#cornerPoints = contour_angle(frame, c)

			if((sd.detect(c) == "rectangle") or (sd.detect(c) == "square")):
				#[extLeft, extRight, extTop, extBottom]

				# Top left (Blue)
				#frame = cv2.circle(frame, (cornerPoints[0][0], cornerPoints[0][1]), 5, (0, 0, 0))
				# Top right (Red)
				#frame = cv2.circle(frame, (cornerPoints[1][0], cornerPoints[1][1]), 5, (0, 0, 0))
				# Bottom right (white)
				#frame = cv2.circle(frame, (cornerPoints[2][0], cornerPoints[2][1]), 5, (0, 0, 0))
				# Bottom left (yellow)
				#frame = cv2.circle(frame, (cornerPoints[3][0], cornerPoints[3][1]), 5, (0, 0, 0))

				frame = cv2.circle(frame, (cPoints[0][0], cPoints[0][1]), 5, (0, 0, 255))
				# Top right (Red)
				frame = cv2.circle(frame, (cPoints[1][0], cPoints[1][1]), 5, (0, 0, 255))
				# Bottom right (white)
				frame = cv2.circle(frame, (cPoints[2][0], cPoints[2][1]), 5, (0, 0, 255))
				# Bottom left (yellow)
				frame = cv2.circle(frame, (cPoints[3][0], cPoints[3][1]), 5, (0, 0, 255))

			#	bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
			#	zyx = np.array([0])
			#	final = np.array([0])
			
			try:
				_, rvec, tvec = cv2.solvePnP(obj_points, np.array(cPoints), zero_camera_matrix, zero_distort_matrix)
				print('tvec', tvec)
				zyx, _ = cv2.Rodrigues(rvec)
				
				reprojectdst, _ = cv2.projectPoints(reprojectsrc, rvec, tvec, zero_camera_matrix,zero_distort_matrix)
				reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
				
				pose_mat = cv2.hconcat((zyx, tvec))
				
				_, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
				
				final = np.concatenate((zyx, tvec), axis=1)
				final = np.concatenate((final, bottom), axis=0)
				final_pose = np.linalg.inv(final)

				try:
					for start, end in line_pairs:
						cv2.line(frame, reprojectdst[start2] / 2, reprojectdst[end] / 2, (0, 0, 255))
				except Exception as e:
					print("nope", e)
					#hello = "no"
			except Exception as e:
				print("no", e)
				#hello = "nope"

		calc_distance, calc_angle1, calc_angle2 = compute_output_values(rvec, tvec)
		
		print(calc_distance, (calc_angle1 * (180 / math.pi)))
		
		average = average * frame_count
		frame_count += 1
		average += calc_distance
		average = average / frame_count

		print("Distance: " + str(calc_distance))
		print("average distance: " + str(average))
		nwTables.putNumber('Distance', calc_distance)
		print("Angle1: " + str(calc_angle1))
		nwTables.putNumber('Angle1', calc_angle1)


		# show the frame to our screen
		cv2.imshow("frame", frame)
		if calc_distance > 60:
			cv2.imwrite("wtf.jpg", frame)
		key = cv2.waitKey(1) & 0xFF
		#print("running")

		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break

		# if the 'q' key is pressed, stop the loop
		if key == ord("a"):
			print("")
			print('euler ang', euler_angle) #, 'points', cornerPoints)

		if key == ord("p"):
			print("")
			print("pose", final_pose)

		if key == ord("d"):
			print("")
			print('distance', distance_maybe, 'angle1', angle1_maybe, 'angle2', angle2_maybe)	#'angle1', (angle1_maybe * (180 / 3.14)), 'angle2', (angle2_maybe * (180 / 3.14)))	

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()

print("okay")
