import cv2
import numpy
import os.path
import math
import glob
import pickle

checkerboard_height = 9
checkerboard_width = 6

square_size = 1.0

shape = None

images = glob.glob('*.jpg')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = numpy.zeros((checkerboard_height * checkerboard_width, 3), numpy.float32)
objp[:, :2] = numpy.mgrid[0:checkerboard_width, 0:checkerboard_height].T.reshape(-1, 2)

# calibrate coordinates to the physical size of the square
objp *= square_size

# Arrays to store object points and image points from all the images.
objpoints = []          # 3d point in real world space
imgpoints = []          # 2d points in image plane.

for fname in images:
    print('Processing file', fname)
    img = cv2.imread(fname)

    if img is None:
        print("ERROR: Unable to read file", fname)
        continue
    shape = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (checkerboard_width, checkerboard_height), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (checkerboard_width, checkerboard_height), corners2, ret)
        cv2.imshow('img', img)

        cv2.waitKey(500)
    else:
        print(fname, 'failed')

cv2.destroyAllWindows()

if not objpoints:
    print("No useful images. Quitting...")

print('Found {} useful images'.format(len(objpoints)))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx)
print(dist)

pickle.dump(mtx, open("mtx.pkl", "wb"))
pickle.dump(dist, open("dist.pkl", "wb"))
