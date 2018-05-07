import cv2
import numpy as np
from matplotlib import pyplot as plt
#import image
image = cv2.imread('test_image.jpg')
#cv2.imshow('orig',image)
#cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)
cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('second',thresh)
#cv2.waitKey(0)

#dilation
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=3)

#cv2.namedWindow('dilated', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('dilated', 1200,600)
#cv2.imshow('dilated',img_dilation)
#cv2.waitKey(0)

#plt.hist(img_dilation.ravel(),256,[0,256]); plt.show()

#find contours
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
#sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
sorted_ctrs = ctrs
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    if (h < 50):
        continue
    
    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    #cv2.imshow('segment no:'+str(i),roi)
    cv2.imwrite('raides/'+str(i)+'.jpg', cv2.resize(cv2.bitwise_not(roi), (28, 28)))
    #cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    #cv2.waitKey(0)

#cv2.imshow('dilated',image)
#cv2.imwrite('folderis/tesktas.jpg', cv2.bitwise_not(image))
cv2.waitKey(0)