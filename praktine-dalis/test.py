# Mute tensorflow debugging information on console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2

from scipy.misc import imsave, imread, imresize
import numpy as np
import argparse
from keras.models import model_from_yaml
import re
import base64
import pickle
import heapq

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0

	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))

	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def load_model(bin_dir):
    # load YAML and create model
    yaml_file = open('%s/model.yaml' % bin_dir, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model

def predict(image_name):
    
    # read parsed image back in 8-bit, black and white mode (L)
    x = imread(image_name, mode='L')
    x = imresize(x,(28,28))
    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)
    # Convert type to float32
    x = x.astype('float32')
    # Normalize to prevent issues with model
    x /= 255
    # Predict from model
    out = model.predict(x)
    # Generate response
    r1 = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]), 'confidence': str(max(out[0]) * 100)[:6]}
    #print(r1)
    return chr(mapping[(int(np.argmax(out, axis=1)[0]))])

# Parse optional arguments
parser = argparse.ArgumentParser(description='Model test')
parser.add_argument('--file', type=str, default='image.png', help='Test file')
args = parser.parse_args()

# Overhead
model = load_model('bin')
mapping = pickle.load(open('bin/mapping.p', 'rb'))

image = cv2.imread(args.file)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#dilation
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=3)
#find contours
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
#sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
sorted_ctrs, _ = sort_contours(ctrs)
result = ''
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    if (h < 50):
        continue
    
    # Getting ROI
    roi = image[y:y+h, x:x+w]
    cv2.imwrite('raides/'+str(i)+'.jpg', cv2.resize(cv2.bitwise_not(roi), (28, 28)))
    result = result + predict('raides/'+str(i)+'.jpg')

print(result)


