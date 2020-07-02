# Imports
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


# Parameters

BS = 1

width_range = [-200, 200]
fill_mode = "nearest"

height_range = 0.5 # 50% of the height of image

rotation_range = 90 # rotate up to 90 degrees

brightness_range = [0.2, 1.0]

zoom_range = [0.5, 1.0]

# Load image
img = load_img("bird.jpg")


# Construct the batch
img = img_to_array(img)

sample = np.expand_dims(img, axis = 0)


# Width shift Augmentation and Showing

"""Instead of commenting use global flags"""

# dataGenerator = ImageDataGenerator(width_shift_range = width_range)
#
# it = dataGenerator.flow(sample, batch_size= BS)
#
# for i in range(9):
#
#     plt.subplot(3, 3, i+1)
#     plt.axis("off")
#     batch = it.next()
#     image = batch[0].astype("uint8")
#     plt.imshow(image)
# plt.show()


# Height shift Augmentation and Showing

"""Instead of commenting use global flags"""

# dataGenerator = ImageDataGenerator(height_shift_range = height_range)
#
# it = dataGenerator.flow(sample, batch_size=BS)
#
# for i in range(9):
#
#     plt.subplot(3, 3, i+1)
#     plt.axis("off")
#     batch = it.next()
#     image = batch[0].astype("uint8")
#     plt.imshow(image)
# plt.show()

# Horizontal flip Augmentation and Showing

# """Instead of commenting use global flags"""
#
# dataGenerator = ImageDataGenerator(horizontal_flip = True)
#
# it = dataGenerator.flow(sample, batch_size=BS)
#
# for i in range(9):
#
#     plt.subplot(3, 3, i+1)
#     plt.axis("off")
#     batch = it.next()
#     image = batch[0].astype("uint8")
#     plt.imshow(image)
# plt.show()

# Random rotation Augmentation and Showing

"""Instead of commenting use global flags"""

# dataGenerator = ImageDataGenerator(rotation_range = rotation_range)
#
# it = dataGenerator.flow(sample, batch_size=BS)
#
# for i in range(9):
#
#     plt.subplot(3, 3, i+1)
#     plt.axis("off")
#     batch = it.next()
#     image = batch[0].astype("uint8")
#     plt.imshow(image)
# plt.show()


# Brightness change Augmentation and Showing

"""Instead of commenting use global flags"""

# dataGenerator = ImageDataGenerator(brightness_range = brightness_range)
#
# it = dataGenerator.flow(sample, batch_size=BS)
#
# for i in range(9):
#
#     plt.subplot(3, 3, i+1)
#     plt.axis("off")
#     batch = it.next()
#     image = batch[0].astype("uint8")
#     plt.imshow(image)
# plt.show()


# Zoom range Augmentation and Showing

"""Instead of commenting use global flags"""

dataGenerator = ImageDataGenerator(zoom_range = zoom_range)

it = dataGenerator.flow(sample, batch_size=BS)

for i in range(9):

    plt.subplot(3, 3, i+1)
    plt.axis("off")
    batch = it.next()
    image = batch[0].astype("uint8")
    plt.imshow(image)
plt.show()