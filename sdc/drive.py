import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras import backend as K

from . import process
from . import model
from . import config

K.set_image_dim_ordering("tf")


sio = socketio.Server()
app = Flask(__name__)

prev_image_array = None

model = model.SteerRegressionModel(input_shape=config.image_size, model=config.model_name)
model.restore(config.model_prefix)

process_image = config.processors["CenterImage"]

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    img_str = data["image"]
    #open("test.jpg", "wb").write(BytesIO(base64.b64decode(img_str)).read())
    img_bytes = BytesIO(base64.b64decode(img_str))
    # process image
    img = process_image(img_bytes)
    steering_angle = model.predict_single(img)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.25
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Remote Driving')
    # parser.add_argument('model', type=str,
    # help='Path to model definition json. Model weights should be on the same path.')
    # args = parser.parse_args()
    # with open(args.model, 'r') as jfile:
    #     model = model_from_json(json.load(jfile))

    # model.compile("adam", "mse")
    # weights_file = args.model.replace('json', 'h5')
    # model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
