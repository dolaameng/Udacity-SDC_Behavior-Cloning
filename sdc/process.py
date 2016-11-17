import cv2
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils


def yuv_normalizer(image_size):
	h, w, nch = image_size
	def fn(img_io):
		"""img_io: either file name or BytesIO"""
		# load image from IO or file
		img = load_img(img_io, target_size=(h, w))
		# convert to uint array
		img_arr = np.asarray(img, dtype=np.uint8)
		# convert to YUV
		yuv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2YUV)
		# normalize
		x = (yuv_img / 255.).astype(np.float32)
		return x
	return fn

def vgg_processor(image_size):
	h, w, nch = image_size
	def fn(img_io):
		"""img_io: either file name or BytesIO"""
		# load image from IO or file
		img = load_img(img_io, target_size=(h, w))
		# convert to uint array
		img_arr = img_to_array(img)
		# convert to YUV
		img_batch = np.expand_dims(img_arr, axis=0)
		# normalize
		x = imagenet_utils.preprocess_input(img_batch)[0]
		return x
	return fn

def rgb_processor(image_size):
	h, w, nch = image_size
	def fn(img_io):
		"""img_io: either file name or BytesIO"""
		# load image from IO or file
		img = load_img(img_io, target_size=(h, w))
		# convert to uint array
		img_arr = img_to_array(img)
		return img_arr
	return fn

# process_single_image = yuv_normalizer(config.image_size)
# process_batch_images = lambda img_ios: np.stack(map(process_single_image, img_ios), axis=0)