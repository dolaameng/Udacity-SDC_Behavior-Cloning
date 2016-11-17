import pytest

from sdc import data
from sdc import process
from sdc import config

import numpy as np


dataset = data.DataSet([("data/t1r1/driving_log.csv", "data/t1r1/IMG")
        , ("data/t1r2/driving_log.csv", "data/t1r2/IMG/")
        , ("data/t1r3/driving_log.csv", "data/t1r3/IMG/")
        , ("data/t1r4/driving_log.csv", "data/t1r4/IMG/")
        , ("data/t1rr1/driving_log.csv", "data/t1rr1/IMG/")])

process_single_image = process.yuv_normalizer(config.image_size)

def test_shuffle():
	N1 = dataset.size()
	dataset.shuffle()
	N2 = dataset.size()
	assert N1 == N2


def test_split():
	train_set, test_set = dataset.split(test_size = 10000)
	assert test_set.size() == 10000
	assert test_set.size() + train_set.size() == dataset.size()
	assert test_set.index == train_set.index == 0

def test_next():
	dataset.reset()
	assert dataset.index == 0
	for i, r in enumerate(dataset):
		assert 1+i == dataset.index
		if i >= 5: break

def test_batch_generator():
	dataset.reset()
	batch_generator = dataset.make_batch_generator(batch_size=64,
							col_grps=[["CenterImage", "LeftImage", "RightImage"],
									"SteeringAngle"])
	ibatch = 0
	while ibatch <= 3:
		images, steers = next(batch_generator)
		assert images.shape == (64, 3)
		assert steers.shape == (64, )
		assert (ibatch+1)*64 == dataset.index
		ibatch += 1


def test_batch_generator_with_process():
	
	dataset.reset()
	batch_generator = dataset.make_batch_generator(batch_size=64,
							col_grps=[["CenterImage", "LeftImage"],
									"Speed",
									"SteeringAngle"],
							process_fns = {"CenterImage": process_single_image,
										   "LeftImage": process_single_image,
										   "RightImage": process_single_image})
	ibatch = 0
	while ibatch <= 3:
		images, speeds, steers = next(batch_generator)
		assert images.shape == (64, 2) + config.image_size
		assert speeds.shape == (64,)
		assert steers.shape == (64,)
		assert 0. <= images.min() <= images.max() <= 1.
		assert -1 <= steers.min() <= steers.max() <= 1.
		ibatch += 1

def test_batch_generator_with_process_single_image():
	dataset.reset()
	batch_generator = dataset.make_batch_generator(batch_size=64,
							col_grps=["CenterImage", "SteeringAngle"],
							process_fns = {"CenterImage": process_single_image})
	ibatch = 0
	while ibatch <= 3:
		xys = next(batch_generator)
		assert type(xys) == tuple
		xs, ys = xys
		assert xs.shape == (64, ) + config.image_size
		assert ys.shape == (64,)
		assert type(xs) == np.ndarray
		ibatch += 1


if __name__ == "__main__":
	pytest.main([__file__])