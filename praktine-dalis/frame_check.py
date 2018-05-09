from __future__ import division
import os
import cv2
import argparse
import pickle
import numpy as np
from scipy.misc import imsave, imread, imresize
from keras.models import model_from_yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    print(r1)
    

# Parse optional arguments
parser = argparse.ArgumentParser(description='Model test')
parser.add_argument('--file', type=str, default='image.png', help='Test file')
args = parser.parse_args()

# Overhead
model = load_model('bin')
mapping = pickle.load(open('bin/mapping.p', 'rb'))

img = cv2.imread(args.file)
width = img.shape[1]	#current image's width
height = img.shape[0]	#current image's height 

print "width : %d\theight : %d"%(width,height)
r = 28/img.shape[0]#aspect_ration
print "ratio: " + str(r)
#ie., here we knows the new images's height so we have to keep the aspect ratio and find new image's width
dim = (int(r*img.shape[1]),28)
resized = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)

for x in range(resized.shape[1]-28):
    crop_img = resized[0:27, x:x+27] # Crop from {x, y, w, h } => {y:h, x:w}
    cv2.imwrite('raides/'+str(x)+'.jpg', cv2.bitwise_not(crop_img))    
    predict('raides/'+str(x)+'.jpg')
