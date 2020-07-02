"""Imports"""
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import load_img, save_img, img_to_array
import keras.backend as K
import numpy as np
import scipy.ndimage
import cv2


"""Dynamic and Hyper parameters"""
layers_contribution = {
    "mixed2": 0.2,
    "mixed3": 3.0,
    "mixed4": 2.0,
    "mixed5": 1.5}

# layers_contribution = {"conv2d_52": 25.0}

lr = 0.01
epochs = 20
octave_scale = 1.4
n_octave = 3

max_loss = 10.
# max_loss = 100.

image_path = "sky.png"
save_path = "result.png"


"""Any kind of process carried on image"""
def preprocess(image_path):

    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def deprocess(img):

    img = np.copy(img)
    img = img.reshape((img.shape[1], img.shape[2], 3))
    img /= 2
    img += 0.5
    img *= 255

    img = np.clip(img, 0, 255).astype("uint8")

    return img
def resize(img, size):

    img = np.copy(img)

    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order = 1)

"""Model configuration"""
model = InceptionV3(weights = "imagenet", include_top = False)
layer_dict = {layer.name: layer for layer in model.layers}

"""Loss configuration"""
loss = K.variable(0.0)
for layer_name, coeff in layers_contribution.items():
    outputs = layer_dict[layer_name].output
    scale = K.cast(K.prod(K.shape(outputs)), "float32")
    loss += coeff * K.sum(K.square(outputs[:, 2:-2, 2:-2, :])) / scale

"""Gradients configuration"""
dream = model.input
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

fetch_loss_grads = K.function([dream], [loss, grads])

def eval_loss_grads(x):
    loss_value, grad_values = fetch_loss_grads([x])
    return loss_value, grad_values


def gradient_ascent(img, iterations, lr,  max_loss = None):

    for i in range(iterations):
        loss_value, grad_values = eval_loss_grads(img)
        print("iteration:{} loss:{}".format(i, loss_value))
        if max_loss is not None and loss_value > max_loss:
            break
        img += lr * grad_values
    return img


"""Shaping"""

img = preprocess(image_path)
successive_shape = [img.shape[1:-1]]

for i in range(1, n_octave):
    shape = tuple([int(dim / (octave_scale ** i) )for dim in img.shape[1:-1]])
    successive_shape.append(shape)

successive_shape.reverse()
original_img = np.copy(img)
shrunk_img = resize(img, successive_shape[0])
"""Train and save result"""
for shape in successive_shape:
    print("Processing shape:{}".format(shape))

    img = resize(img, shape)
    img = gradient_ascent(img, epochs, lr, max_loss)

    downscale_img = resize(original_img, shape)
    upscale_img = resize(shrunk_img, shape)
    lost_details = downscale_img - upscale_img
    img += lost_details

    shrunk_img = resize(original_img, shape)
save_img(save_path, deprocess(img))


"""Main condition"""
# Not needed