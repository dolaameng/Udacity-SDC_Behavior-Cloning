"""
Configuration of preprocessing, SDC model and training process, e.g.,

- `train_data` to specify multiple training folders
- `model_prefix` to specify where the trained model to be saved
- turn on one model implementation by comment on its configuration part and comment off others - yeah, not ugly, but convinient for exploration
"""

from . import process


## common settings
batch_size = 256
model_prefix = "models/model"
xycols = ["CenterImage", "SteeringAngle"]
train_data = [
          ("data/t1r1/driving_log.csv", "data/t1r1/IMG")
        , ("data/t1r2/driving_log.csv", "data/t1r2/IMG/")
        , ("data/t1r3/driving_log.csv", "data/t1r3/IMG/")
        , ("data/t1r4/driving_log.csv", "data/t1r4/IMG/")
        , ("data/t1r5/driving_log.csv", "data/t1r5/IMG/")
        # clean data has too many small steering, which is actually bad
        #, ("data/t1c1/driving_log.csv", "data/t1c1/IMG/")
        # bridge section is repeated because it is short
        , ("data/t1b1/driving_log.csv", "data/t1b1/IMG/")
        # , ("data/t1b2/driving_log.csv", "data/t1b2/IMG/") 
        # a "little" wiggle helps for VGG model - misbehaviors will be corrected
        , ("data/t1w1/driving_log.csv", "data/t1w1/IMG/") 
        # training by reverse driving, to get more right turns
        # or you can mirror the image and minus the steer
        , ("data/t1rr1/driving_log.csv", "data/t1rr1/IMG/")

        , ("data/t2r1/driving_log.csv", "data/t2r1/IMG/")
        , ("data/t2r2/driving_log.csv", "data/t2r2/IMG/")
        , ("data/t2r3/driving_log.csv", "data/t2r3/IMG/")
        , ("data/t2r3.1/driving_log.csv", "data/t2r3.1/IMG/")
        , ("data/t2r4/driving_log.csv", "data/t2r4/IMG/")
        , ("data/t2w1/driving_log.csv", "data/t2w1/IMG/")
]


# # nvidia model
# # 1. the default normalization used by conv2d requires input to be in range (0, 1)
# model_name="nvidia"
# image_size = (80, 160, 3)
# processors = {"CenterImage": process.yuv_normalizer(image_size)}


# vgg 16 setting
# 1. looks like a square size e.g. (80, 80) is consistently better than (80, 160)
model_name = "vgg16_pretrained"
image_size = (80, 80, 3)
processors = {"CenterImage": process.vgg_processor(image_size)}

# # vgg 16 multilayer setting
# # 1. use 80 x 80
# model_name = "vgg16_multi_layers"
# image_size = (80, 80, 3)
# processors = {"CenterImage": process.vgg_processor(image_size)}


# # comma ai setting
# model_name = "comma_ai"
# image_size = (160, 320, 3)
# processors = {"CenterImage": process.rgb_processor(image_size)}