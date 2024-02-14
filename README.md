# CS545-NightTime-Detection
Goal: achieve the same edge/corner detection performance on a night time image, compared to its daytime counterpart. 

Steps:
1. Apply Canny Edge Detection/Harris Corner Detection to the daytime image with default parameters to produce a ground-truth edge/corner map image.
2. Apply image processing (or other methods/algorithms you prefer) to process the night-time image.
3. Apply the same detection method to the processed image (not necessarily with default parameters) to output another image.
4. Try to make the output image as close as possible to the ground-truth image. We use MSE as the metric to measure the error/difference.
