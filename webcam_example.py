import sys
import time
import torch
import torch.nn as nn
import cv2
from torch.autograd import Variable
import numpy as np
from skimage.feature import peak_local_max
from skimage.transform import resize as sk_resize

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

def intersects(rect1, rect2):
    x1, y1, w1, h1, _ = rect1
    x2, y2, w2, h2, _ = rect2
    x_intersection = False
    y_intersection = False
    if x1 <= x2 + w2 and x1 + w1 >= x2:
        x_intersection = True
    if x2 <= x1 + w1 and x2 + w2 >= x1:
        x_intersection = True
    if y1 <= y2 + h2 and y1 + h1 >= y2:
        y_intersection = True
    if y2 <= y1 + h1 and y2 + h2 >= y1:
        y_intersection = True

    if x_intersection and y_intersection:
        return True
    else:
        return False


def detect_img(model, img):
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

	output = model(Variable(img, volatile=True))

	output = output.data.numpy()

	output = output[0, 0, :, :]
	return output

def detect_multi_scale(model, img, scales, recepive_field):
	result = []
	for scale in scales:
			scaled_img = cv2.resize(img, (int(img.shape[1]/scale), int(img.shape[0]/scale)))
			output = detect_img(model, np.float32(scaled_img))
			output = sk_resize(output, output_shape=img.shape[:2], preserve_range=True)
			output = np.float32(output)
			#cv2.imshow("Scale " + str(scale), output)
			#cv2.waitKey(10)
			coordinates = peak_local_max(output, min_distance=30, threshold_abs=-1)
			for y, x in coordinates:
				detection = map(int, [x, y, recepive_field*scale, recepive_field*scale, output[y, x]])
				result.append(detection)

	# Non maximal surpression:
	result = sorted(result, key = lambda r: r[-1], reverse=True)
	after_nms = []
	for detection in result:
		keep_detection = True
		for other_detection in after_nms:
			if intersects(detection, other_detection):
				keep_detection = False
				break
		if keep_detection:
			after_nms.append(detection)

	return after_nms

def detect_webcam(model):
	recepive_field_size, offset = recepive_field(model)
	print "Receptive field", recepive_field_size, "Center offset", offset
	scales = [2.7, 3.5]
	cap = cv2.VideoCapture(0)

	while True:
		coordinates = []
		ret, img = cap.read()
		t0 = time.time()
		coordinates = detect_multi_scale(model, img, scales, recepive_field_size)
		t1 = time.time()
		print("Inference time for frame", t1 - t0)


		for x, y, width, height, score in coordinates:
			x = x - offset//2
			y = y - offset//2
			cv2.rectangle(img, (x-width//2, y-height//2), (x+width//2, y+height//2), (0, 255, 0))
		cv2.imshow("result", img)
		cv2.waitKey(10)

if __name__ == '__main__':
	model = get_model(sys.argv[1])
	detect_webcam(model)