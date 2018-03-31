import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET

def get_model(path):
	layers = []
	in_channels = 3

	tree = ET.parse(path)
	elems = [elem for elem in tree.iter()]
	elems = elems[::-1]

	for elem in elems:
		meta = elem.attrib
		if elem.tag == 'con' or elem.tag == 'affine_con':
			for key in meta:
				meta[key] = int(meta[key])
			data = ''.join([x for x in elem.text.strip() \
				if x != '\n' and x != '\r'])
			data = [float(x) for x in data.split(' ') if x != '']
			data = np.float32(data)

		if elem.tag == 'con':
			layer = torch.nn.Conv2d(in_channels=in_channels,
			        out_channels=meta['num_filters'],
			        kernel_size = (meta['nr'], meta['nc']),
			        stride = (meta['stride_y'], meta['stride_x']),
			        padding = (meta['padding_y'], meta['padding_x']),
			        bias = True)


			size = meta['num_filters']*meta['nr']*meta['nc']*in_channels

			filter_data = data[ : size]
			bias_data = data[size : ]

			filter_data = np.reshape(filter_data, (meta['num_filters'], \
				in_channels, meta['nr'], meta['nc']))

			layer.weight.data = torch.from_numpy(filter_data)
			layer.bias.data = torch.from_numpy(bias_data)

			layers.append(layer)
			in_channels = meta['num_filters']

			# The dilation can be increased to have a larger receptive field, 
			# for example for use in webcams when the person is 
			# close to the camera.
			layer.dilation = 1

		
		elif elem.tag == 'affine_con':
			layer = torch.nn.BatchNorm2d(num_features=in_channels)

			layer.running_var = \
				torch.from_numpy(np.float32(np.ones(in_channels)))
			layer.running_mean = \
				torch.from_numpy(np.float32(np.zeros(in_channels)))

			layer.weight.data = torch.from_numpy(data[ : in_channels])
			layer.bias.data = torch.from_numpy(data[in_channels : ])

			layers.append(layer)
			
		elif elem.tag == 'relu':
			layers.append(nn.ReLU())

	model = nn.Sequential(*layers)
	return model