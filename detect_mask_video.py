from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pafy
import sys

displayfacebox = False
displayeyebox = False
displaycolourcircle = False
displayfill = False

def findIfMask(frame, faceNet, maskNet):
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []


	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:

		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model")
ap.add_argument("-c", "--confidence", type=float, default=0.5)
args = vars(ap.parse_args())


prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


maskNet = load_model(args["model"])


print("Starting Video")
vs = VideoStream(src=0).start()
time.sleep(2.0)

url = "https://www.youtube.com/watch?v=FtutLA63Cp8"
video = pafy.new(url)

streams = video.streams
for s in streams:
	print(s)

best = video.getbest(preftype="mp4")
vid = cv2.VideoCapture(best.url)

lower = np.array([103,74,49]) 
upper = np.array([163,134,109])

while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	
	try:
		(locs, preds) = findIfMask(frame, faceNet, maskNet)
	except:
		print("test")
		pass

	for (box, pred) in zip(locs, preds):
		
		 
		(startX, startY, endX, endY) = box
		x = startX
		y = startY + (endY - startY) // 5
		w = endX - startX
		h = (endY - startY) // 3

		ret, vidframe = vid.read()
		ret, vidframe = vid.read()
		ret, vidframe = vid.read()
		if (x+int(w/2)<frame.shape[1] and y+int(1.5*h)<frame.shape[0]):
			pass
		else:
			continue
		(mask, withoutMask) = pred

		mask1 = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
		mask1[y+int(0.5*h):y+int(3*h),x:x+w]=1


		if(mask > withoutMask):
			mask2 = np.zeros((frame.shape[0]+2,frame.shape[1]+2), dtype=np.uint8)
			mask2.fill(1)
			mask2[y+int(0.5*h):y+int(3*h),x:x+w]=0
			vidframe = cv2.resize(vidframe,(frame.shape[1],frame.shape[0]),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
			cv2.floodFill(frame, mask2, seedPoint=(x+int(w/2),y+int(1.5*h)), newVal=(0,0,0), loDiff=(3,3,3,3), upDiff=(3,3,3,3), flags=8)
			
			uwu = np.zeros((frame.shape[0],frame.shape[1],3), dtype=np.uint8)
			y1 = y+int(0.5*h)
			h1 = int(2.5*h)
			if (y1+h1>=frame.shape[0]):
				continue
			vidframe = cv2.resize(vidframe,(w,h1),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
			uwu[y1:y1+h1,x:x+w]+=vidframe
			vidframe=uwu
			if (displayfill):
				frame = vidframe + frame

		if(displayeyebox):
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
		if(displaycolourcircle):
			cv2.circle(frame, (x+int(w/2),y+int(1.5*h)), 10, (0, 255, 0))		

		label = "Mask Detected" if mask > withoutMask else "Please put on a mask"
		color = (0, 255, 0) if label == "Mask Detected" else (0, 0, 255)

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)


		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		if( displayfacebox):
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	if key == ord("q"):
		break
	if key == ord("v"):
		displayfill ^= True
	if key == ord("b"):
		displayfacebox ^= True
	if key == ord("n"):
		displayeyebox ^= True
	if key == ord("m"):
		displaycolourcircle ^= True

cv2.destroyAllWindows()
vs.stop()
