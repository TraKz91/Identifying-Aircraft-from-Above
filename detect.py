# import the necessary packages
#code inspire from https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import config
# take the image and parse it
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

#load the model and label
mod = load_model(config.modpath)
labl = pickle.loads(open(config.labpath, "rb").read())
# load the input image
img = cv2.imread(args["image"])
img = imutils.resize(img, width=1000)
# run selective search
select= cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
select.setBaseImage(img)
select.switchToSelectiveSearchFast()
rectangle = select.process()
#create two list
prop = []
carre = []
i=0
#run through each bounding box and add the region to prop and the position for carre
for (x, y, w, h) in rectangle[:config.maximum_prop_inference]:

	region = img[y:y + h, x:x + w]
	region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
	region = cv2.resize(region, config.set_dim,
		interpolation=cv2.INTER_CUBIC)

	#cv2.imshow("tadaronne"+str(i),roi)
	region = img_to_array(region)
	region = preprocess_input(region)
	prop.append(region)
	carre.append((x, y, x + w, y + h))
	i+=1
# convert the proposals and bounding boxes into NumPy arrays
prop = np.array(prop, dtype="float32")
carres = np.array( carre, dtype="int32")
prop=prop/255
probabilite = mod.predict(prop) #see for each prop, it is an aircraft or not
print(probabilite)
#assign the write label
labels = labl.classes_[np.argmax(probabilite, axis=1)]
indexs = np.where(labels == 0)[0]

carres = carres[indexs]
probabilite = probabilite[indexs][:, 1] #extract the aircraft from the bounding boxes

idxs = np.where(probabilite >= config.Proba_minimum)#if the probability is met, it is an aircraft
carres= carres[indexs]
probabilite = probabilite[indexs]

img1 = img.copy()
#draw a bounding box around the true positive
for (carre, probabilite) in zip(carres, probabilite):
	(debX, debY, finX, finY) = boite
	cv2.rectangle(img1, (debX, debY), (finX, finY),
		(0, 255, 0), 2)
	y = debY - 10 if debY- 10 > 10 else debY + 10
	text= "Aircraft: {:.2f}%".format(probabilite * 100)
	cv2.putText(clone, text, (debX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
cv2.imshow("Classification", img1)
cv2.waitKey(0)
