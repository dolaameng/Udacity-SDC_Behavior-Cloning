"""
This is the main script to train the SDCRegression model implemented in `model.py`

Usage:

1. To build a model from scratch, run
```cmd
python -m sdc.train_regression --train --nb_epoch 6
```
where `sdc.train_regression` build a vgg16-based regression model from CenterImage to Steer. `--nb_epoch` is the number of training epoches.

2. To continuously train a model from a previous one, run
```cmd
python -m sdc.train_regression --train --restore --nb_epoch 6
```

3. To load a trained model to evaluate on test data without any training, simply run
```cmd
python -m sdc.train_regression --restore
```
"""

import numpy as np

import argparse
import tensorflow as tf
import keras.backend as K
import pandas as pd

from . import model
from . import config
from . import process
from . import data

from .config import xycols, processors


# parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--restore", dest="restore", action="store_true")
parser.add_argument("--nb_epoch", dest="nb_epoch", type=int)
args = parser.parse_args()

nb_epoch = args.nb_epoch

# load training data
dataset = data.DataSet(config.train_data).preprocess()
# split to train, validation and test
trainset, others = dataset.split(test_size=20000)
valset, testset = others.split(test_size=10000)

train_size = trainset.size()
val_size = valset.size()
test_size = testset.size()

print("training size %d, validation size %d, test size %d" % (train_size, val_size, test_size))

# make batch_generator out of the corresonponding datasets
train_generator = trainset.make_batch_generator(batch_size=config.batch_size, 
						col_grps = xycols,
						process_fns = processors)
val_generator = valset.make_batch_generator(batch_size=config.batch_size,
						col_grps = xycols,
						process_fns = processors)
test_generator = testset.make_batch_generator(batch_size=config.batch_size,
						col_grps = xycols,
						process_fns = processors)

# training the model
with tf.Session() as sess:
	# create the model
	K.set_session(sess)
	img_steer_model = model.SteerRegressionModel(input_shape=config.image_size, model=config.model_name)
	# model information
	print("inspect model:")
	img_steer_model.inspect()
	# restore from previous training if instructed to do so
	if args.restore:
		print("restore model from", config.model_prefix)
		img_steer_model.restore(config.model_prefix)

	if args.train:
		img_steer_model.fit_generator(nb_epoch, train_generator, 
			train_size//config.batch_size*config.batch_size, 
			val_generator, val_size)
		# save the model
		print("save model to", config.model_prefix)
		img_steer_model.save(config.model_prefix)
	# evaluation on test data for future inspection
	print("evaluate on test data")
	testset.reset()
	test_y = testset.log.SteeringAngle
	test_yhat = img_steer_model.predict_generator(test_generator, test_size)
	mse = np.mean((test_yhat - test_y) * (test_yhat - test_y))
	print("test mse:", mse)
	# save the test result
	print("save test result for inspection")
	test_result = pd.DataFrame({"steer": test_y, 
								"prediction": test_yhat, 
								"image": testset.log.CenterImage})
	test_result.to_csv("tmp/test_result.csv", header=True, index=False)