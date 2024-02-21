import cv2 
import numpy as np 
from det_metrics import mse

# load night and day images
day_img_path = "./images/daytime.jpg"
night_img_path = "./images/nighttime.jpg"
day_img = cv2.imread(day_img_path)
night_img = cv2.imread(night_img_path)

# load low-light enhanced and denoised nighttime image
enh_night_img_path = "./SwinIR/results/swinir_color_dn_noise25/nighttime_enhanced_SwinIR.png"
enh_night_img_orig = cv2.imread(enh_night_img_path)
enh_night_img = cv2.imread(enh_night_img_path, cv2.IMREAD_GRAYSCALE)

# create a copy of the image object
day_corner_map = day_img.copy()
night_corner_map = night_img.copy()
enh_night_corner_map = enh_night_img_orig.copy()
  
# convert the input image into grayscale color space 
day_img = np.float32(cv2.cvtColor(day_img, cv2.COLOR_BGR2GRAY)) 
night_img = np.float32(cv2.cvtColor(night_img, cv2.COLOR_BGR2GRAY)) 
enh_night_img_orig = cv2.cvtColor(enh_night_img_orig, cv2.COLOR_BGR2GRAY)

# perform histogram equalization on enhanced night image
enh_night_img = cv2.equalizeHist(enh_night_img)
  
# apply the cv2.cornerHarris method 
day_corners = cv2.cornerHarris(day_img, 2, 5, 0.07) 
night_corners = cv2.cornerHarris(night_img, 2, 5, 0.07) 
enh_night_corners = cv2.cornerHarris(enh_night_img, 2, 5, 0.07) 
  
# Results are marked through the dilated corners 
day_corners = cv2.dilate(day_corners, None) 
night_corners = cv2.dilate(night_corners, None)
enh_night_corners = cv2.dilate(enh_night_corners, None)
  
# Reverting back to the original image with optimal threshold value 
day_corner_map[day_corners > 0.01 * day_corners.max()] = [0, 0, 255] 
night_corner_map[night_corners > 0.01 * night_corners.max()] = [0, 0, 255] 
enh_night_corner_map[enh_night_corners > 0.01 * enh_night_corners.max()] = [0, 0, 255] 

# threshold corner maps
day_corners[day_corners <= 0.01 * day_corners.max()] = 255
night_corners[night_corners <= 0.01 * night_corners.max()] = 255
enh_night_corners[enh_night_corners <= 0.01 * enh_night_corners.max()] = 255
day_corners = np.pad(day_corners, [(0,0), (0,1)], mode='constant', constant_values=0) # adjust width to match night image for MS

# compute MSE between groundtruth and initial prediction
mse_initial = mse(day_corners, night_corners)
print(f"MSE for non-enhanced nighttime edge map: {mse_initial}")

# compute MSE between groundtruth and final prediction
mse_final = mse(day_corners, enh_night_corners)
print(f"MSE for enhanced nighttime edge map: {mse_final}")
  
# the window showing output image with corners 
cv2.imshow('Daytime Harris Corner Map', day_corners) 
cv2.imshow('Non-enhanced Nighttime Harris Corner Map', night_corners) 
cv2.imshow('Enhanced Nighttime Harris Corner Map', enh_night_corners) 

cv2.imshow('Enh Nighttime Image', enh_night_img) 
cv2.imshow('Enh Night Image No Eq', enh_night_img_orig)
cv2.waitKey(0) 
cv2.destroyAllWindows() 