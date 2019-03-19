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

vs = None
out = None

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	vs = VideoStream(src="/dev/visioncam").start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

ts = time.now()

print('starting...')

while True:
	print("#################################################################################################")

	frame_height, frame_width = frame.shape[:2]
	try:
		if out is None:
			out = cv2.VideoWriter("/media/nvidia/logs/output-{}.avi".format(ts), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
	except:
		out = None
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	if frame is None:
		continue

		if out is not None:
			out.write(frame)


# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()
	if out is not None:
		out.release()


print("done...")