import pytest
import numpy as np

from sdc import model
from sdc import config
from sdc import process
from sdc import data

dataset = data.DataSet([("data/t1r1/driving_log.csv", "data/t1r1/IMG")])

trainset, others = dataset.split(test_size=0.6)
valset, testset = others.split(test_size=0.5)

train_size = trainset.size()
val_size = valset.size()
test_size = testset.size()

train_generator = trainset.make_batch_generator(batch_size=config.batch_size,
						col_grps = ["CenterImage", "SteeringAngle"],
						process_fns = {"CenterImage": process.process_single_image})
val_generator = valset.make_batch_generator(batch_size=config.batch_size,
						col_grps = ["CenterImage", "SteeringAngle"],
						process_fns = {"CenterImage": process.process_single_image})
test_generator = testset.make_batch_generator(batch_size=config.batch_size,
						col_grps = ["CenterImage", "SteeringAngle"],
						process_fns = {"CenterImage": process.process_single_image})

def test_model_training():
	img_steer_model = model.SteerRegressionModel(input_shape=config.image_size)
	assert img_steer_model.input_shape == config.image_size

	nb_epoch = 2
	img_steer_model.fit_generator(nb_epoch, train_generator, train_size, val_generator, val_size)
	test_yhat = img_steer_model.predict_generator(test_generator, test_size)
	mse = np.mean((test_yhat - testset.log.SteeringAngle) * (test_yhat - testset.log.SteeringAngle))
	assert test_yhat.shape == testset.log.SteeringAngle.shape
	assert mse <= 0.1

def test_predict_single_image():
	img_steer_model = model.SteerRegressionModel(input_shape=config.image_size, model="vgg16")

	nb_epoch = 1
	img_steer_model.fit_generator(nb_epoch, train_generator, train_size, val_generator, val_size)
	x = process.process_single_image("test.jpg")
	test_yhat = img_steer_model.predict_single(x)
	assert -1.5 <= test_yhat <= 1.5


if __name__ == "__main__":
	pytest.main([__file__])