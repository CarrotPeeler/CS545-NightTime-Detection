# CS545-NightTime-Detection
Goal: achieve the same edge/corner detection performance on a night time image, compared to its daytime counterpart. 

Proceedure Overview:
1. Apply Canny Edge Detection/Harris Corner Detection to the daytime image with default parameters to produce a ground-truth edge/corner map image.
2. Apply image processing (or other methods/algorithms you prefer) to process the night-time image.
3. Apply the same detection method to the processed image (not necessarily with default parameters) to output another image.
4. Try to make the output image as close as possible to the ground-truth image. We use MSE as the metric to measure the error/difference.

## Setup
Download the SwinIR weights for color denoising ([005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth](https://github.com/JingyunLiang/SwinIR/releases)). Place the weights in SwinIR/model_zoo/swinir.

## Method
### Running low-light image enhancment using Zero-DCE
```
python Zero-DCE_code/lowlight_test.py
```
Place the enhanced image from the images folder in SwinIR/testsets/nighttime.

### Running colored image denoising using SwinIR
```
cd SwinIR
python main_test_swinir.py --task color_dn --noise 25 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth --folder_gt testsets/nighttime
```

### Running Canny/Harris Detector
Canny Edge Detection
```
python canny_detector.py
```
Harris Corner Detection
```
python harris_detector.py
```
