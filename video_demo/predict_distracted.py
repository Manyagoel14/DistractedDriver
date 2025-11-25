# import the necessary packages
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from driver_prediction import predict_result

INPUT_VIDEO_FILE = "input_video.mp4"
OUTPUT_VIDEO_FILE = "output_video.mp4"

vs = cv2.VideoCapture(INPUT_VIDEO_FILE)
writer = None
(W, H) = (None, None)
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	# clone the output frame, then convert it from BGR to RGB ordering, resize the frame to a fixed 224x224, and then perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224))
	# frame -= mean
	frame = np.expand_dims(frame,axis=0).astype('float32')

	# make predictions on the frame and then update the predictions queue
	label = predict_result(frame)

	# draw the activity on the output frame
	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, 30,
			(W, H), True)
	# write the output frame to disk
	writer.write(output)
	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
