import cv2 

day_img_path = "./images/daytime.jpg"
night_img_path = "./images/nighttime.jpg"

day_img = cv2.imread(day_img_path)  # Read image 
night_img = cv2.imread(night_img_path)  # Read image 
  
# Setting parameter values 
t_lower_default = 50  # Lower Threshold 
t_upper_default = 150  # Upper threshold
t_lower_night = 50  # Lower Threshold 
t_upper_night = 150  # Upper threshold 
  
# Applying the Canny Edge filter 
day_edge_map = cv2.Canny(day_img, t_lower_default, t_upper_default) 
night_edge_map = cv2.Canny(night_img, t_lower_default, t_upper_default) 
night_edge_map_v2 = cv2.Canny(night_img, t_lower_night, t_upper_night) 

# cv2.imshow('Original Image', day_img) 
# cv2.imshow('Daytime Edge Map', day_edge_map) 
# cv2.imshow('Nighttime Edge Map', night_edge_map) 
# cv2.imshow('Nighttime Edge Map V2', night_edge_map_v2) 

img = cv2.imread(night_img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img,5)

img_th = cv2.adaptiveThreshold(img,
                               maxValue=255,
                               adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               thresholdType=cv2.THRESH_BINARY,
                               blockSize=13,
                               C=1)
cv2.imshow("th", img_th)
cv2.waitKey(0) 
cv2.destroyAllWindows() 