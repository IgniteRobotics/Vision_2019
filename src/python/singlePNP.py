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

# obstructed test
greenLower = (0,73,22) 
greenUpper = (90,255,78)

# The (x,y,z) points for the corners of the vision target, in the order top left, top right, bottom right, bottom left
# these are "square" to the floor.  i.e. the long edge of the tape is vertical
# this is a lie, but may be useful to calculate if the tape is right or left side.
#obj_points = np.array([[0, 0, 0], [2, 0, 0], [2, 5.75, 0], [0, 5.75, 0]], np.float32)
obj_points = np.array([[0, 0, 0], [1.945, -0.467, 0], [0.605, -6.058, 0], [-1.34, -5.591, 0]], np.float32)
#TODO: this is totally wrong!
top_obj_points = np.array([[0, 0, 0], [11.89, 0, 0], [9.945, -0.467, 0], [1.945, -0.467, 0]], np.float32)
obj_points = top_obj_points

# paths to the cameraMatrix and distortMatrix files
cameraMatrix_filepath = "cameraMatrix.pkl"
distortMatrix_filepath = "distortMatrix.pkl"

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

	# epsilon = 0.1*cv2.arcLength(contour,True)
	# approx = cv2.approxPolyDP(contour,epsilon,True)
	# print('===== approx =====')
	# print(approx)


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
	print ('===== corners =====')
	print (to_ret)
	return to_ret


axis = np.float32([[6,0,0], [0,6,0], [0,0,6]]).reshape(-1,3)						   

def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)
    return img

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
# def find_closest_by_Y(candidates):
#     # most distant y factors:
# 	#y_values = list(map(lambda pt: pt[0][1] , candiates))
# 	#delta = max(y_values), min(y_values)
# 	points = candidates[0], candidates[1]
# 	for i, element in enumerate(candidates):
# 		for j, sec_element in enumerate(candidates):
# 			print(element, sec_element)
# 			if i == j:
# 				continue
# 			if abs(sec_element[0][0][1] - element[0][0][1]) < abs(points[0][0][1] - points[1][0][1]):
# 			 	points = sec_element, element
# 	fourpoints = sec_element[0], sec_element[1], element[0], element[1]
# 	return np.int0(cv2.boxPoints(fourpoints))

def find_highest_Y_Pts(candidates):

    to_ret = []
    for i in range(2):
        minY = math.inf
        minPt = []
        minYIdx = 0
        for i, element in enumerate(candidates):
            if (element[0][1] < minY):
                minY = element[0][1]	
                minPt = element
                minYIdx = i        
        print("length:",len(candidates))
        del(candidates[minYIdx])
        to_ret.append(minPt[0])
        to_ret.append(minPt[1])
        #to_ret.append(np.array([minPt[0], minPt[1]]))
        
    print(np.array(to_ret))
    return np.array(to_ret)

def slope(x1, y1, x2, y2):
	return np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    #return (y2-y1)/(x2-x1)

def pickTapePairs(contours, img):
    candidates = []
    for contour in contours:
        #print('==== contour ====')
        perimeter = cv2.arcLength(contour,True)
        epsilon = 0.01*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        #print ('length:', len(approx))
        area = cv2.contourArea(contour)
        #print ('area:', area)
        if area < 500:
            print("skipping cnt w/ area: ", area)
            continue 
        # find minimum area
        rect = cv2.minAreaRect(contour)
        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.int0(box)
        #print('box: ',box)
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
        if topSlope > 90:
            topSlope = 180 - topSlope
        #if topSlope > 17 or topSlope < 10:  #13.5 +/- 3.5 deg
        #    continue
        
        print('==== contour ====')		
        print('highestPt:', minYIndex, highestPt[0],highestPt[1])
        print('nextHighestPt:', nextHighestPt[0],nextHighestPt[1])
        print('slope:', topSlope)
        candidates.append((highestPt, nextHighestPt))
        # draw contours
        print(type(box))
        print(box)
        cv2.drawContours(img, [box], 0, (0,0, 255), 3)
        #print(contour)
    print('================')
    print(candidates)

    top_contour = find_highest_Y_Pts(candidates)
    print("top_contour", top_contour)
    cv2.drawContours(img, [top_contour], -1, (0,255, 0), 3)

    return top_contour, img




################# MAIN LOOP ###################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", help="path to the image file")
#args = vars(ap.parse_args())

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

#frame = None

#if args.get("image", False):
#	frame = cv2.imread(args.get("image"))
#else:
#	print("no file to read!")
#	exit(-1)

# allow the camera or video file to warm up
time.sleep(2.0)

while True:
    # grab the current frame
    frame = vs.read()

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
    final_pose = None
    reprojectdst = None

    frame_height, frame_width = frame.shape[:2] 

    corners = [np.array([0, 0]), np.array([frame_width, 0]), np.array([frame_width, frame_height]), np.array([0, frame_height])]

    mid_frame = (int(frame_width / 2), int(frame_height / 2))

    frame_hsv = rid_noise(frame)

    # find contours in thresholded frame, then grab the largest one
    cnts = cv2.findContours(frame_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("count of contours:", len(cnts))
    # draw them all
    # cv2.drawContours(frame, cnts, -1, (0, 255, 0), 2)

    cTop = None
    c = None

    if len(cnts) > 0:
        cTop, frame = pickTapePairs(cnts, frame)
        c = cTop

    # c = None
    # # Get the biggest contour
    # c = cnts[0]
    # for contour in cnts:
    # 	if contour_comparator(contour, c):
    # 		c = contour

    if c is not None and len(c) != 0:
        # cornerPoints = contour_angle(frame, c)

        # M = cv2.moments(c)
        # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # cv2.circle(frame, center, 5, (0, 0, 255), -1)
        #cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        

        # find the corners of the contour
        cornerPoints = get_corners(corners, c)

        print ('corners', cornerPoints)

        cpTop = get_corners(corners, cTop)
        print ('top countour corners', cpTop)
        

        #determine side
        side = None
        if (cornerPoints[0][0] > cornerPoints[3][0]): #top left is to the right, so it's the left tape.
            side = "LEFT"
        else:
            side = "RIGHT"

        # Top left (Blue)
        #cv2.circle(frame, (cornerPoints[0][0], cornerPoints[0][1]), 5, (255, 0, 0))
        # Top right (Red)
        #cv2.circle(frame, (cornerPoints[1][0], cornerPoints[1][1]), 5, (0, 0, 255))
        # Bottom right (white)
        #cv2.circle(frame, (cornerPoints[2][0], cornerPoints[2][1]), 5, (255, 255, 255))
        # Bottom left (yellow)
        #cv2.circle(frame, (cornerPoints[3][0], cornerPoints[3][1]), 5, (0, 255, 255))

        #initialize some arrays
        rot = np.array([0])
        
        #_, rvec, tvec = cv2.solvePnP(obj_points, np.array(cornerPoints), cameraMatrix, distortMatrix)
        _, rvec, tvec = cv2.solvePnP(cTop, np.array(cornerPoints), zero_camera_matrix, zero_distort_matrix)
        
        print('===== tvec =====')
        print(tvec)
        print('===== rvec =====')
        print(rvec)
        

        rot, jac = cv2.Rodrigues(rvec)
        print('===== rot =====')
        print(rot)


        # turn it into euler angles
        euler_angles = rotationMatrixToEulerAngles(rot)
        print('===== eulers =====')
        print(list(map(lambda x: x * (180 / math.pi), euler_angles)))
        #axispts, jac = cv2.projectPoints(axis, rvec, tvec, cameraMatrix,distortMatrix)
        axispts, jac = cv2.projectPoints(axis, rvec, tvec, zero_camera_matrix,zero_distort_matrix)
        print ('===== axispts =====')
        print(axispts)

        #add the axis
        #frame = draw_axis(frame,cornerPoints,axispts)

        #get the angle to the contour
        # what does this get you?  I DON'T KNOW
        #t = (math.asin(-rot[0][2]))

        #get individual positions.
        x = tvec[0][0]
        y = tvec[1][0]
        z = tvec[2][0]

        distance = math.sqrt(x**2 + z**2)

        # find angles
        ang_bot_to_target = math.atan2(x, z)

        inv_tvec = tvec
        inv_tvec[1][0] = -tvec[1][0]
        inv_tvec[2][0] = -tvec[2][0]
        print ('======= inv_tvec =====')
        print (inv_tvec)

        #inverse of rot mat
        rot_inv = rot.transpose()
        #p0_world = np.matmul(rot_inv, -tvec)
        p0_world = np.matmul(rot_inv, inv_tvec)
        print('===== p0_world =====')
        print(p0_world)

        # rodrigues works in both directions.
        inv_rvec, jac = cv2.Rodrigues(rot_inv)
        print ('===== inv_rvec =====')
        print (inv_rvec)
        
        # I don't think any of this is right.
        #p0_axispts, jac = cv2.projectPoints(axis, inv_rvec, -tvec, zero_camera_matrix,zero_distort_matrix)
        #print('===== p0_ap =====')
        #print(p0_axispts)
        # frame = draw_axis(frame,cornerPoints,p0_axispts)
        
        angle_target_to_bot = math.atan2(p0_world[0][0], p0_world[2][0])
        angle_t2 = math.atan2(p0_world[0][0], p0_world[1][0])
        angle_t3 = math.atan2(p0_world[1][0], p0_world[2][0])

        print('posed angles:', angle_target_to_bot*180/math.pi, angle_t2*180/math.pi, angle_t3*180/math.pi)

        print("x", x, "y", y, "z", z)
        print("angle from bot to target", (ang_bot_to_target*180/math.pi))
        print("angle from target to bot", (angle_target_to_bot*180/math.pi))
        print("distance", distance)

        cv2.putText(frame, "Distance: " + "{:7.2f}".format(distance), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)
        cv2.putText(frame, "Heading: " + "{:7.2f}".format((ang_bot_to_target*180/math.pi)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)
        cv2.putText(frame, "Side: " + "{0}".format(side), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)


    cv2.imshow("frame", frame)
    #cv2.imwrite("out.jpg", frame)
    key = cv2.waitKey(1) & 0xFF

if key == ord("q"):
	exit(1)
	cv2.destroyAllWindows()
	print("okay")