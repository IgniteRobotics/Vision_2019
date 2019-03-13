# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import cv2
from imutils import contours
from imutils import perspective
import numpy as np
import argparse
import imutils
 
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
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
    print('dist inputs:', tl[np.newaxis], ':',rightMost)
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    print('Dist:', D)

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--new", type=int, default=-1,
	help="whether or not the new order points should should be used")
args = vars(ap.parse_args())
 
# load our input image, convert it to grayscale, and blur it slightly
image = cv2.imread("target2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
 
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
 
# sort the contours from left-to-right and initialize the bounding box
# point colors
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

# loop over the contours individually
for (i, c) in enumerate(cnts):
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 100:
        continue

    # compute the rotated bounding box of the contour, then
    # draw the contours
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

    # show the original coordinates
    print("Object #{}:".format(i + 1))
    print(box)

    # rect = perspective.order_points(box)
    rect = order_points(box)

    # show the re-ordered coordinates
    print(rect.astype("int"))
    print("")

    # loop over the original points and draw them
    for ((x, y), color) in zip(rect, colors):
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    # draw the object num at the top-left corner
    cv2.putText(image, "Object #{}".format(i + 1),
        (int(rect[0][0] - 15), int(rect[0][1] - 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)