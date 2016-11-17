import pytest

from sdc import process
from sdc import config
import numpy as np

def test_yuv_normalizer():
	process_image = process.yuv_normalizer(image_size=(80, 80, 3))
	img = process_image("data/t1r1/IMG/center_2016_11_07_21_25_00_341.jpg")
	assert img.shape == (80, 80, 3)
	assert img.dtype == np.float32
	assert 0. <= img.min() < img.max() <= 1.

def test_vgg_processor():
	process_image = process.yuv_normalizer(image_size=(80, 80, 3))
	img = process_image("data/t1r1/IMG/center_2016_11_07_21_25_00_341.jpg")
	assert img.shape == (80, 80, 3)
	assert img.dtype == np.float32
	assert -255. <= img.min() < img.max() <= 255.

def _test_process_single_image():
	process_image = process.yuv_normalizer(image_size=config.image_size)
	img_path = "data/t1r1/IMG/center_2016_11_07_21_25_00_341.jpg"
	img1 = process_image(img_path)
	img2 = process.process_single_image(img_path)
	assert np.all(img1==img2)

def _test_process_batch_images():
	img_paths = np.array(["data/t1r1/IMG/center_2016_11_07_21_25_00_341.jpg"
				, "data/t1r1/IMG/center_2016_11_07_21_25_00_410.jpg"
				, "data/t1r1/IMG/center_2016_11_07_21_25_00_497.jpg"
				, "data/t1r1/IMG/center_2016_11_07_21_25_00_504.jpg"
				, "data/t1r1/IMG/center_2016_11_07_21_25_00_555.jpg"])
	imgs = process.process_batch_images(img_paths)
	assert imgs.shape == (5, ) + config.image_size

if __name__ == "__main__":
	pytest.main([__file__])