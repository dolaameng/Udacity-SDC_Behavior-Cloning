"""This implements several image processing methods for different models.
The common interface to the methods:
- `image_size` parameter specifies the target shape of outputs
- each processor takes an `img_io` (a image file path or BytesIO str) as input
and generates the processed image as numpy.array.
- the `img_io` param might have additional instructions on how to convert the image, e.g., 
a suffix "_mirror" in image file name indicates the image should be mirrored 
horizontally.
"""


import cv2
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils


def yuv_normalizer(image_size):
	"""Load image, reshape to image_size, 
	convert to YUV channel, and normalize pixels to [0, 1]
	"""
	h, w, nch = image_size
	def fn(img_io):
		"""img_io: either file name or BytesIO"""
		# load image from IO or file
		if type(img_io) == str and img_io.endswith("_mirror"):
			img_io = img_io[:-7]
			ismirror = True
		else:
			ismirror = False
		img = load_img(img_io, target_size=(h, w))
		# convert to uint array
		img_arr = np.asarray(img, dtype=np.uint8)
		# convert to YUV
		yuv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2YUV)
		# normalize
		x = (yuv_img / 255.).astype(np.float32)
		if ismirror:
			x = x[:, ::-1, :]
		return x
	return fn

def vgg_processor(image_size):
	"""Load image, reshape to image_size,
	convert to BGR channel and normalize pixels by subtracting predefined means.
	"""
	h, w, nch = image_size
	def fn(img_io):
		"""img_io: either file name or BytesIO"""
		# load image from IO or file
		if type(img_io) == str and img_io.endswith("_mirror"):
			img_io = img_io[:-7]
			ismirror = True
		else:
			ismirror = False
		img = load_img(img_io, target_size=(h, w))
		# convert to uint array
		img_arr = img_to_array(img)
		img_batch = np.expand_dims(img_arr, axis=0)
		# normalize
		x = imagenet_utils.preprocess_input(img_batch)[0]
		if ismirror:
			x = x[:, ::-1, :]
		return x
	return fn

def rgb_processor(image_size):
	"""Load image, reshape to image_size,
	without any further preprocessing to the images.
	"""
	h, w, nch = image_size
	def fn(img_io):
		"""img_io: either file name or BytesIO"""
		# load image from IO or file
		if type(img_io) == str and img_io.endswith("_mirror"):
			img_io = img_io[:-7]
			ismirror = True
		else:
			ismirror = False
		img = load_img(img_io, target_size=(h, w))
		# convert to uint array
		img_arr = img_to_array(img)
		if ismirror:
			img_arr = img_arr[:, ::-1, :]
		return img_arr
	return fn
