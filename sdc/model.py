"""
This implements the SDC model, with a convenient interfaces for testing and driving scripts.
It delegates the training and prediction to the Keras model underneath.

There are different implementations for the underneath model, choose the one to explore in
`config.py`
"""


import numpy as np
import json

from keras.applications import VGG16
from keras.layers import AveragePooling2D, Conv2D
from keras.layers import Input, Flatten, Dense, Lambda, merge
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam
from keras.models import Model, Sequential, model_from_json
from keras.regularizers import l2
from keras import backend as K

# fix the image ordering for tensorflow
K.set_image_dim_ordering("tf")

class SteerRegressionModel(object):
	def __init__(self, input_shape, model="vgg16_pretrained"):
		"""
		input_shape: it is the image shape, e.g., (80, 80, 3) for single image regression.
		model: name of implemented models, {"nvidia", "vgg16_pretrained", "vgg16_multi_layers", "comma_ai"}
		"""
		self.input_shape = input_shape
		self.model = None
		np.random.seed(1337)
		self.model_name = model
		self.build()

	def build(self):
		"""build the architecture of underlying model
		"""
		try:
			build_fn = getattr(self, "_build_"+self.model_name)
		except:
			raise ValueError("model %s not implemented" % self.model_name)

		print ("building %s model" % self.model_name)
		build_fn()
		return self

	def _build_nvidia(self):
		"""Model based on Nvidia paper https://arxiv.org/pdf/1604.07316v1.pdf
		"""
		inp = Input(shape=self.input_shape)

		x = Conv2D(24, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu")(inp)
		x = Conv2D(36, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu")(x)
		x = Conv2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu")(x)
		x = Conv2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu")(x)
		x = Conv2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu")(x)
		x = Flatten()(x)
		## using dropout tends to make all-zero predictions! 
		x = Dense(1164, activation="elu")(x)
		#x = Dropout(0.8)(x)
		x = Dense(100, activation="elu")(x)
		#x = Dropout(0.8)(x)
		x = Dense(50, activation="elu")(x)
		#x = Dropout(0.8)(x)
		x = Dense(10, activation="elu")(x)
		#x = Dropout(0.8)(x)
		x = Dense(1, activation="linear")(x)

		self.model = Model(input=inp, output=x)
		return self
	
	def _build_vgg16_pretrained(self):
		"""Pretrained VGG16 model with fine-tunable last two layers
		"""
		input_image = Input(shape = self.input_shape)

		base_model = VGG16(input_tensor=input_image, include_top=False)
		
		for layer in base_model.layers[:-3]:
		    layer.trainable = False

		W_regularizer = l2(0.01)

		x = base_model.get_layer("block5_conv3").output
		x = AveragePooling2D((2, 2))(x)
		x = Dropout(0.5)(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Flatten()(x)
		x = Dense(4096, activation="elu", W_regularizer=l2(0.01))(x)
		x = Dropout(0.5)(x)
		x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
		x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
		x = Dense(1, activation="linear")(x)


		self.model = Model(input=input_image, output=x)
		return self

	def _build_vgg16_multi_layers(self):
		"""Use outputs from multiple vgg layers as input to steering modelling
		Without normalization, the result is not really good
		"""
		input_image = Input(shape = self.input_shape)

		
		base_model = VGG16(input_tensor=input_image, include_top=False)
		
		for layer in base_model.layers:
		    layer.trainable = False

		x3 = base_model.get_layer("block5_conv1").output
		x3 = AveragePooling2D((2, 2))(x3)
		x3 = Flatten()(x3)
		x4 = base_model.get_layer("block5_conv2").output
		x4 = AveragePooling2D((2, 2))(x4)
		x4 = Flatten()(x4)
		x5 = base_model.get_layer("block5_conv3").output
		x5 = AveragePooling2D((2, 2))(x5)
		x5 = Flatten()(x5)

		x = merge([x3, x4, x5], mode="concat", concat_axis=1)

		x = Dropout(0.5)(x)
		x = Dense(4096, activation="elu")(x)
		x = Dropout(0.5)(x) # tend to predict all zeros!
		x = Dense(2048, activation="elu")(x)
		x = Dense(1024, activation="elu")(x)
		x = Dense(1, activation="linear")(x)


		self.model = Model(input=input_image, output=x)
		return self

	def _build_comma_ai(self):
		""" Example model from https://github.com/commaai/research
		"""
		model = Sequential()
		model.add(Lambda(lambda x: x/127.5 - 1.,
						input_shape=self.input_shape,
						output_shape=self.input_shape))
		model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
		model.add(ELU())
		model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
		model.add(ELU())
		model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
		model.add(Flatten())
		model.add(Dropout(.2))
		model.add(ELU())
		model.add(Dense(512))
		model.add(Dropout(.5))
		model.add(ELU())
		model.add(Dense(1))

		self.model = model
		return self

	def inspect(self):
		"""Inspect the underlying model by layer name and parameters
		"""
		for layer in self.model.layers:
			trainable = None
			# merge layer of keras doesn't have trainable?
			try: 
				trainable = layer.trainable
			except:
				pass
			print(layer.name, layer.input_shape, layer.output_shape, trainable)
		# print(self.model.summary())
		return self

	def fit_generator(self, nb_epoch,
				train_generator, train_size,
				val_generator, val_size, 
				loss_fn=None, optimizer=None):
		"""train the model by feeding a train and valiation batch_generator.
		Params:
			- nb_epoch: # of training epoch
			- train_generator: data generator for training
			- train_size: # of training examples in the train_generator
			- val_generator: data generator for validation
			- val_size: # of examples in validation set
			- loss_fn: default "mse"
			- optimizer: default Adam with learning rate 0.001
		"""
		self.loss_fn = loss_fn or "mse"
		optimizer = optimizer or Adam(lr=0.001)
		self.model.compile(loss=self.loss_fn, optimizer=optimizer)
		self.history = self.model.fit_generator(train_generator,
							samples_per_epoch=train_size,
							nb_epoch=nb_epoch,
							validation_data=val_generator,
							nb_val_samples=val_size)
		return self

	def predict_generator(self, test_generator, test_size):
		"""moddel prediction on a batch generator of test dataset
		"""
		i = 0
		yhats = []
		while i <= test_size:
			bxy = next(test_generator)
			if len(bxy) == 2:
				bx, by = bxy
			else:
				bx = bxy
			yhats.append(self.model.predict_on_batch(bx)[:, 0])
			i += bx.shape[0]

		yhat = np.hstack(yhats)[:test_size]
		return yhat

	def predict_single(self, x):
		"""predict the output based on a single input, e.g.,
		x is a single image, it returns a single steering value.
		"""
		y = self.model.predict(np.expand_dims(x, axis=0))[0][0]
		return y

	def save(self, prefix):
		"""save model for future inspection and continuous training
		"""
		model_file = prefix + ".json"
		weight_file = prefix + ".h5"
		json.dump(self.model.to_json(), open(model_file, "w"))
		self.model.save_weights(weight_file)
		return self

	def restore(self, prefix):
		"""restore a saved model
		"""
		model_file = prefix + ".json"
		weight_file = prefix + ".h5"
		self.model = model_from_json(json.load(open(model_file)))
		self.model.load_weights(weight_file)
		return self
