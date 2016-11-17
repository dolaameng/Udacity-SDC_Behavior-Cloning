from . import process

# original image_size = (160, 320, 3)

## common settings
batch_size = 128
model_prefix = "models/model"
xycols = ["CenterImage", "SteeringAngle"]


# # nvidia model
# # 1. the default normalization used by conv2d requires input to be in range (0, 1)
# model_name="nvidia"
# image_size = (80, 160, 3)
# processors = {"CenterImage": process.yuv_normalizer(image_size)}


# # vgg 16 setting
# # 1. looks like a square size e.g. (80, 80) is consistently better than maintaining shapes (80, 160)
# model_name = "vgg16"
# image_size = (80, 80, 3)
# processors = {"CenterImage": process.vgg_processor(image_size)}

# comma ai setting
model_name = "comma.ai"
image_size = (160, 320, 3)
processors = {"CenterImage": process.rgb_processor(image_size)}