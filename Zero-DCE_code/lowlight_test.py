import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
 
def lowlight(image_path, savepath, ckpt_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight)/255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load(ckpt_path))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	print("End time: ", end_time)
	torchvision.utils.save_image(enhanced_image, savepath)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		root_dir = os.getcwd() + "/images"
		img_path = root_dir + "/nighttime.jpg"
		savepath = root_dir + "/nighttime_enhanced.jpg"
		ckpt_path = os.getcwd() + "/Zero-DCE_code\snapshots\Epoch99.pth"
		print(img_path)
		lowlight(img_path, savepath, ckpt_path)

		# filePath = 'data/test_data/'
		# file_list = os.listdir(filePath)
		# for file_name in file_list:
		# 	test_list = glob.glob(filePath+file_name+"/*") 
		# 	for image in test_list:
		# 		# image = image
		# 		print(image)
		# 		lowlight(image)

		

