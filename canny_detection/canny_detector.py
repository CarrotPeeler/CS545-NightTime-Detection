import cv2 
import numpy as np
from cv2.ximgproc import anisotropicDiffusion

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

day_img_path = "./images/daytime.jpg"
night_img_path = "./images/nighttime.jpg"

day_img = cv2.imread(day_img_path)  # Read image 
night_img = cv2.imread(night_img_path)  # Read image 
  
# Setting parameter values 
t_lower_default = 50  # Lower Threshold 
t_upper_default = 150  # Upper threshold
  
# Applying the Canny Edge filter 
# day_edge_map = cv2.Canny(day_img, t_lower_default, t_upper_default) 
night_edge_map = cv2.Canny(night_img, t_lower_default, t_upper_default) 
# night_edge_map_v2 = cv2.Canny(night_img, t_lower_night, t_upper_night) 

# cv2.imshow('Original Image', day_img) 
# cv2.imshow('Daytime Edge Map', day_edge_map) 
# cv2.imshow('Nighttime Edge Map', night_edge_map) 
# cv2.imshow('Nighttime Edge Map V2', night_edge_map_v2) 


# cv2.imshow("th", img_th)

# night_img_invert = cv2.bitwise_not(night_img)
# cv2.imshow("Inverted Night", night_img_invert)

# n_inc = increase_brightness(night_img, value=40)
# n_inc = cv2.cvtColor(night_img, cv2.COLOR_BGR2GRAY) 
enh_night_img_path = "./images/nighttime_enhanced.jpg"
enh_night_img = cv2.imread(enh_night_img_path)
# kernel = np.ones((3,3), np.float32)/9
# enh_night_img = cv2.filter2D(enh_night_img, -1, kernel)
# enh_night_img_d = anisotropicDiffusion(enh_night_img, 0.075, 80, 100)
# enh_night_img_d = cv2.fastNlMeansDenoisingColored(enh_night_img,
#                                                 dst=None,
#                                                 h=10,
#                                                 hColor=10,
#                                                 templateWindowSize=7,
#                                                 searchWindowSize=21)
# enh_night_img_d = cv2.imread(enh_night_img_path, cv2.IMREAD_GRAYSCALE)

enh_night_img_d = cv2.imread(enh_night_img_path, cv2.IMREAD_GRAYSCALE)

# kernel = np.ones((3,3), np.float32)/9
# enh_night_img_d = cv2.filter2D(enh_night_img, -1, kernel)
# enh_night_img_d = cv2.cvtColor(enh_night_img_d, cv2.COLOR_BGR2GRAY)
enh_night_img_d = cv2.GaussianBlur(enh_night_img_d, (7,7), 0)
enh_night_img_d = cv2.equalizeHist(enh_night_img_d)

# enh_night_img_d = cv2.adaptiveThreshold(enh_night_img_d,
#                                maxValue=255,
#                                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                thresholdType=cv2.THRESH_BINARY,
#                                blockSize=11,
#                                C=3)
t_lower_night = 50  # Lower Threshold 
t_upper_night = 150  # Upper threshold 
n_edge_map = cv2.Canny(enh_night_img_d, t_lower_night, t_upper_night) 

# cv2.imshow('night', night_img) 
cv2.imshow('enhanced night denoised', enh_night_img_d)
# cv2.imshow('enhanced night orig', enh_night_img) 
cv2.imshow('mod edge Map', n_edge_map) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 