import sys
import time
import torch
import torch.nn as nn
import cv2
from torch.autograd import Variable
import numpy as np
from skimage.feature import peak_local_max
from skimage.transform import resize as sk_resize
from skimage.transform import rotate

from dlib_torch_converter import get_model

def recepive_field(model):
	start_in = 0
	j_in = 1
	r_in = 0
	for module in model._modules:
		m = model._modules[module]
		if type(m) is nn.Conv2d:
			if type(m.dilation) is tuple:
				dilation = m.dilation[0]
			else:
				dilation = m.dilation
			kernel_size = m.kernel_size[0] + (m.kernel_size[0]-1)*(dilation-1)
			j_out = j_in * m.stride[0]
			r_out = r_in + (kernel_size - 1) * m.stride[0]
			start_out = start_in + ((kernel_size-1)/2  - m.padding[0])*j_in

			start_in, j_in, r_in = start_out, j_out, r_out
			
	return r_out, start_out

def detect_img(img, model):
	model.eval()
	orig_shape = img.shape[0 : 2]

	r=122.781998
	g=117.000999
	b=104.297997

	img[: , :, 0] -= b
	img[: , :, 1] -= g
	img[: , :, 2] -= r
	
	img = img[:, :, ::-1].copy() / 256.0

	img = img.transpose((2, 0, 1))
	img = np.float32(img)
	img = torch.from_numpy(img)

	img = img.unsqueeze(0)

	t0 = time.time()
	output = model(Variable(img, volatile=True))
	t1 = time.time()
	print("Inference time for frame", t1 - t0)

	output = output.data.numpy()

	output = output[0, 0, :, :]
	output = sk_resize(output, output_shape=orig_shape, preserve_range=True)
	output = np.float32(output)
	coordinates = peak_local_max(output, min_distance=50, threshold_abs=1)

	cv2.imshow("Raw model output", output)
	cv2.waitKey(10)
	return coordinates

def detect_webcam(model):
	box_size, box_offset = recepive_field(model)
	print "Receptive field", box_size, "Center offset", box_offset
	cap = cv2.VideoCapture(0)
	while True:
		ret, img = cap.read()
		print img.shape
		coordinates = detect_img(np.float32(img), model)

		N = box_size
		for y, x in coordinates:
			x = x - box_offset//2
			y = y - box_offset//2
			cv2.rectangle(img, (x-N, y-N), (x+N, y+N), (0, 255, 0))
		cv2.imshow("result", img)
		cv2.waitKey(10)

if __name__ == '__main__':
	model = get_model(sys.argv[1])
	detect_webcam(model)