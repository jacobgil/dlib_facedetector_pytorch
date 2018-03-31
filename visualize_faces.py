import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import xml.etree.ElementTree as ET
import cv2
import scipy.misc
import random
from skimage.transform import rotate
from dlib_torch_converter import get_model

def rotate_img(img, angle):
    for index in range(3):
        channel = img[0, index, :, :]
        channel = rotate(channel, angle, resize=False, preserve_range=True)
        img[0, index, :, :] = channel
    return np.float32(img)


def tensor_to_np_image(input):
    img = input.cpu().data.numpy()[0, :]
    img = np.transpose(img, (1, 2, 0))
    img = img - np.min(img)
    img = img / np.max(img)
    return img    

def activation_maximization(model, filter_index):
    model.eval()
    N = 128
    input = np.float32(np.ones((1, 3, N, N))) / 255
    input = Variable(torch.from_numpy(input), requires_grad=True)

    # forward  pass:
    iterations = 900
    lr = 1.0

    for i in xrange(iterations):

        img = input.data.numpy()
        # Random rotation:
        angle = np.float32(np.random.uniform(-30, 30)) * 1
        
        use_flip = random.random() > 0.5
        if use_flip:
        	img[:, :, :, :] = img[:, :, :, ::-1]

        img = rotate_img(img, angle)


        input.data = torch.from_numpy(img)
        
        input_cuda = input
        model.zero_grad()

        # The activation is before softmax
        out = model(input_cuda)
        #out = torch.nn.Sigmoid()(out)
        size = out.size(2)
        # Take the middle pixel in the image.
        # For classification this will be just out[0, category]
        loss = out[0, filter_index, size//2, size//2]
        loss.backward()

        # Normalize the gradient to a unit vector
        grad_cpu = input.grad
        grad_cpu = grad_cpu / torch.norm(grad_cpu, 2)
        data = grad_cpu.data.numpy()

        input = input - lr * grad_cpu
        input.volatile=False
        input.requires_grad=True

        data = input.data.numpy()

        img = input.data.numpy()
        img = rotate_img(img, -angle)
        if use_flip:
        	img[:, :, :, :] = img[:, :, :, ::-1]

        # Weight decay regularization for the visualization
        input.data = torch.from_numpy(img)

        print(i)
        img = tensor_to_np_image(input)
        cv2.imshow("img", img)
        cv2.waitKey(10)


    img = tensor_to_np_image(input)
    img = np.uint8(255 * img)
    return img

if __name__ == '__main__':
    model = get_model(sys.argv[1])

    # Skip the last layer (and the BN layer before it)
    model = nn.Sequential(*[model._modules[i] \
        for i in model._modules.keys()[:-2]])

    for filter_index in range(30):
    	for index in range(10):
            img = activation_maximization(model, filter_index)
            output_path = "_".join([str(filter_index), str(index)]) + ".jpg"
            cv2.imwrite(output_path, img)