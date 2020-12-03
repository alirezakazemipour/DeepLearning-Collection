# Imports
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.layers import Input, UpSampling2D, Concatenate
from keras.models import Model
import os
from skimage import io, transform
import numpy as np


# Dynamic and Hyper params
BS = 2
epochs = 4000
dropout_rate = 0.5

model_name = "best_model.h5"
image_size = 256

filters = [64, 128, 256, 512, 1024]
down_conv_buffer = []

# Adjust data
def adjust(img, mask):

    img /= 255
    mask /= 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return img, mask

# Generators configs
def generate_train(aug_dict,
                   train_path,
                   image_folder,
                   mask_folder,
                   batch_size,
                   target_size = (256, 256),
                   save_to_dir = None,
                   img_save_prefix = "img",
                   mask_save_prefix = "mask",
                   seed = 1):

    img_generator = ImageDataGenerator(**aug_dict)
    mask_generator = ImageDataGenerator(**aug_dict)

    imgs = img_generator.flow_from_directory(train_path,
                                             target_size=target_size,
                                             classes=[image_folder],
                                             class_mode=None,
                                             batch_size=batch_size,
                                             color_mode="grayscale",
                                             save_to_dir=save_to_dir,
                                             save_prefix=img_save_prefix,
                                             seed=seed)
    masks = mask_generator.flow_from_directory(train_path,
                                               target_size=target_size,
                                               classes=[mask_folder],
                                               class_mode=None,
                                               batch_size=batch_size,
                                               color_mode="grayscale",
                                               save_to_dir=save_to_dir,
                                               save_prefix=mask_save_prefix,
                                               seed=seed)
    for img, mask in zip(imgs, masks):
        img, mask = adjust(img, mask)
        yield img, mask


def generate_test(path, num_img = 30, imgsize = (image_size, image_size), as_gray = True):
    for i in range(num_img):
        img = io.imread(os.path.join(path, "{}.png".format(i)), as_gray = as_gray)
        img /= 255
        img = transform.resize(img, imgsize)
        img = np.reshape(img, (1,) + img.shape + (1,))
        yield img






gen_args_dict = dict(rotation_range=0.2,
                     width_shift_range = 0.05,
                     height_shift_range = 0.05,
                     zoom_range = 0.05,
                     horizontal_flip = True,
                     shear_range = 0.5,
                     fill_mode = "nearest")
generator = generate_train(aug_dict=gen_args_dict,
                           train_path="./data/membrane/train",
                           batch_size=20,
                           image_folder="image",
                           mask_folder="label",
                           save_to_dir = "./data/membrane/train/aug")
num_batch = 3
for i, batch in enumerate(generator):
    if i >= num_batch:
        break


# Model configs
def down_conv(inputs,
              filter,
              dropout = False,
              max_pooling = True):

    global down_conv_buffer
    x = inputs
    # print("down conv input shape{}".format(x.shape) )

    if max_pooling:
        x = MaxPool2D(strides=(2, 2))(x)
    x = Conv2D(filters=filter,
               kernel_size=(3, 3),
               # strides=(2, 2),
               padding="same",
               kernel_initializer="he_normal",
               activation="relu")(x)

    x = Conv2D(filters=filter,
               kernel_size=(3, 3),
               # strides=(2, 2),
               padding="same",
               kernel_initializer="he_normal",
               activation="relu")(x)
    # print("down conv 2nd conv shape{}".format(x.shape))
    down_conv_buffer.append(x)

    if dropout:
        x = Dropout(rate=dropout_rate)(x)
    outputs = x

    return outputs


def up_conv(inputs,
            y,
            filter):

    x = inputs
    x = UpSampling2D()(x)

    x = Conv2D( filters=filter,
                kernel_size=(2, 2),
                # strides=(2, 2),
                padding="same",
                kernel_initializer="he_normal",
                activation="relu" )( x )

    print("feedforward shape{}, Upsampled shape{}".format(y.shape, x.shape))
    x = Concatenate()([y, x])

    x = Conv2D( filters=filter,
                kernel_size=(3, 3),
                # strides=2,
                padding="same",
                kernel_initializer="he_normal",
                activation="relu" )( x )

    x = Conv2D( filters=filter,
                kernel_size=(3, 3),
                # strides=2,
                padding="same",
                kernel_initializer="he_normal",
                activation="relu" )( x )
    return x


inputs = Input(shape=(image_size, image_size, 1))
x = inputs

for filter in filters:

    if filter == filters[0]:
        x = down_conv(x, filter, max_pooling=False)
    elif filter == filters[-2] or filter == filters[-1]:
        x = down_conv( x, filter, dropout=True )
    else:
        x= down_conv(x, filter)


down_conv_buffer.pop() # Last layer is not concatenated
for filter in filters[-2::-1]:
    x = up_conv(x, down_conv_buffer.pop(), filter)

outputs = Conv2D(filters=1, #Binary classification
           kernel_size=1,
           strides=1,
           padding="same",
           activation="sigmoid")(x)

model = Model(inputs, outputs, name="UNet")
model.summary()

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["acc"])


# Checkpoint peripheral and fittig

generator = generate_train(aug_dict=gen_args_dict,
                           train_path="./data/membrane/train",
                           image_folder="image",
                           mask_folder="label",
                           batch_size=BS)

checkpoint = ModelCheckpoint(filepath=model_name,
                             monitor="val_loss",
                             verbose=1,
                             save_best_only=True)
# Train for 10 epochs

model.fit_generator(generator,
                    steps_per_epoch = epochs // BS,
                    epochs = 10,
                    callbacks = [checkpoint])

# Save model for further use

model.save_weights(model_name)

# Test generator
test_generator = generate_test("./data/membrane/test")

# Load weights and predict
model.load_weights(model_name)
preds = model.predict_generator(test_generator, steps=30, verbose=1)

#Save result
for i, item in preds:
    img = item[:, :, 0]
    io.imsave(os.path.join("/data/membrane/test","{}_predicted.png".format(i)), img)
