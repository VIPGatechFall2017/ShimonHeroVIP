import cv2
import sys
import os, fnmatch

def find(pattern, path):
    for file in os.listdir(path):
        if file.endswith(pattern):
            return os.path.join(path, file)
    return None

def printCapture(vidcap):
	length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps    = vidcap.get(cv2.CAP_PROP_FPS)
	print("Frame Count: " + str(length))
	print("Width: " + str(width))
	print("Height: " + str(height))
	print("FPS: " + str(fps))

path = sys.argv[1]
file = find('.mov', path)
if file:
	vidcap = cv2.VideoCapture(file)

	printCapture(vidcap)

	count = 0
	success = True
	while success:
		success,image = vidcap.read()
		print('Read a new frame: ', success)
		if success:
			cv2.imwrite(os.path.join(path, "images/frame%d.jpg" % count), image)     # save frame as JPEG file
			count += 1