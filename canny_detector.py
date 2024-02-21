import cv2 
import numpy as np
from det_metrics import mse

# Paths to day and night images
day_img_path = "./images/daytime.jpg"
night_img_path = "./images/nighttime.jpg"

# load both images as arrays
day_img = cv2.imread(day_img_path)  # Read image 
night_img = cv2.imread(night_img_path)  # Read image 
  
# Setting parameter values for Canny Detector
t_lower_default = 50  # Lower Threshold 
t_upper_default = 150  # Upper threshold
  
# Applying the Canny Edge filter 
day_edge_map = cv2.Canny(day_img, t_lower_default, t_upper_default) # groundtruth edge map
day_edge_map = np.pad(day_edge_map, [(0,0), (0,1)], mode='constant', constant_values=0) # adjust width to match night image for MSE
night_edge_map = cv2.Canny(night_img, t_lower_default, t_upper_default) # initial prediction

# compute MSE between groundtruth and initial prediction
mse_initial = mse(day_edge_map, night_edge_map)
print(f"MSE for non-enhanced nighttime edge map: {mse_initial}")

# load low-light enhanced and denoised nighttime image
sharp_night_img_path = "./SwinIR/results/swinir_color_dn_noise25/nighttime_enhanced_SwinIR.png"
sharp_night_img = cv2.imread(sharp_night_img_path, cv2.IMREAD_GRAYSCALE)

# perform histogram equalization on enhanced night image
sharp_night_img = cv2.equalizeHist(sharp_night_img)

# Apply lower thresholds, as the nighttime image is generally darker
t_lower_night = 40 # Lower Threshold 
t_upper_night = 100 # Upper threshold 

# generate enhanced edge map
enh_night_edge_map = cv2.Canny(sharp_night_img, t_lower_night, t_upper_night) # final prediction

# compute MSE for enhanced nighttime edge map
mse_final = mse(day_edge_map, enh_night_edge_map)
print(f"MSE for enhanced nighttime edge map: {mse_final}")

# display edge maps
cv2.imshow('Daytime Edge Map', day_edge_map) 
cv2.imshow('Non-Enhanced Nighttime Edge Map', night_edge_map)
cv2.imshow('Enhanced Nighttime Edge Map', enh_night_edge_map) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 