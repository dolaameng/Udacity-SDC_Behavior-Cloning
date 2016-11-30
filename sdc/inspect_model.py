"""Inpsect trained SDC model by looking at:
1. Output of certain conv layers
2. Visualization of filters of certain conv layers
"""

import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # docker setting
import matplotlib.pyplot as plt
from scipy.misc import imresize

import keras.backend as K
from keras.models import Model

from . import process
from . import model
from . import config

parser = argparse.ArgumentParser()
# layer to be visulized
parser.add_argument("--layer", dest="layer", type=str)
# path to input image
parser.add_argument("--image", dest="image", type=str)
args = parser.parse_args()

layer_name = args.layer or "block2_conv2"
image_path = args.image

K.set_image_dim_ordering("tf")

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

with tf.Session() as sess:
    K.set_session(sess)

    # restore keras model
    sdc_model = model.SteerRegressionModel(input_shape=config.image_size, model=config.model_name)
    sdc_model.restore(config.model_prefix)
    train_model = sdc_model.model
    # build a new model to output the certain layer 
    # Ref: https://github.com/fchollet/keras/issues/41
    visual_model = Model(train_model.input, train_model.get_layer(layer_name).output)

    # load image
    process_img = config.processors["CenterImage"]
    img = process_img(image_path)
    print("image shape", img.shape)

    # generate output
    layer_output = visual_model.predict(np.expand_dims(img, 0))[0]
    #layer_output = deprocess_image(layer_output)
    layer_output = (layer_output - layer_output.min()) / (layer_output.max() - layer_output.min())
    print("generating output for layer", layer_name)
    print(layer_output.shape)
    
    i = 25
    output_img = layer_output[:,:,i]
    print(output_img/255.)
    output_img = imresize(output_img, config.image_size[:2])

    output_img = img * np.expand_dims(output_img/255., -1)
    output_img = deprocess_image(output_img)
    plt.imshow(output_img, cmap=plt.cm.gray)
    plt.imsave("a.png", output_img, cmap = plt.cm.gray)