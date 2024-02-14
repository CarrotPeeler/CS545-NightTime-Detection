import cv2 
import numpy as np 

img_path = "./images/daytime.jpg"
img = cv2.imread(img_path) 
# create a copy of the image object
corner_map = img.copy()
  
# convert the input image into 
# grayscale color space 
operated_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
# modify the data type 
# setting to 32-bit floating point 
operated_img = np.float32(operated_img) 
  
# apply the cv2.cornerHarris method 
# to detect the corners with appropriate 
# values as input parameters 
dest = cv2.cornerHarris(operated_img, 2, 5, 0.07) 
  
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
  
# Reverting back to the original image, 
# with optimal threshold value 
corner_map[dest > 0.01 * dest.max()]=[0, 0, 255] 
  
# the window showing output image with corners 
cv2.imshow('Original Image', img) 
cv2.imshow('Harris Corner Map', corner_map) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 