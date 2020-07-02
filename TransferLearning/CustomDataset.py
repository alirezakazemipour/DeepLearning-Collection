# Imports
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Hyper parameters
IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 2

TRAIN_BS = 64
VALID_BS = 64

EARLY_STOPPING_PATIENCE = 5

lr = 0.01
moment = 0.9
decay = 1e-6
epochs = 10

train_dir = "./train"
valid_dir = "./valid"
test_dir = "./test"


weights_name = "best_model.h5"

# Model construction
model = Sequential()

base_model = ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
                      include_top=False,
                      pooling = "avg")

model.add(base_model)
model.add(Dense(NUM_CLASSES, activation = "softmax"))

model.layers[0].trainable = False # Forgot to write in the first place !!!!

model.summary()

# Model Compilation
opt = SGD(lr = lr,
          momentum = moment,
          decay = decay,
          nesterov = True)
model.compile(opt,
              loss="categorical_crossentropy",
              metrics=["acc"])

# Augment training and validation Dataset
dataGenerator = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = dataGenerator.flow_from_directory(train_dir,
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE), # Wrongly and unnecessarily wrote number of channels in last dimension
                                                    classes=NUM_CLASSES,
                                                    batch_size=TRAIN_BS,
                                                    class_mode="categorical") # Forgot to write "class_mode" param in the first place !!!!

valid_generator = dataGenerator.flow_from_directory(directory=valid_dir,
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                    classes=NUM_CLASSES,
                                                    batch_size=VALID_BS,
                                                    class_mode="categorical")

# Adjust fitting peripherals

checkpoint  = ModelCheckpoint(filepath=weights_name,
                              monitor="val_loss",
                              verbose=1,
                              save_best_only=True)
earlyStop = EarlyStopping(verbose=1,
                          monitor="val_loss",
                          patience= EARLY_STOPPING_PATIENCE)

# Train with history supported
if os.path.exists(weights_name):
    model.load_weights(weights_name)

else:
    model_history = model.fit_generator(train_generator,
                                        steps_per_epoch=m_train // TRAIN_BS, # Add m_train: needs Dataset
                                        epochs=epochs,
                                        verbose=1,
                                        callbacks=[checkpoint, earlyStop],
                                        validation_data=valid_generator,
                                        validation_steps=20)


# Show history
plt.figure(1, figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.plot(model_history["loss"])
plt.plot(model_history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model loss")
plt.legend(["train", "valid"])
plt.show()

plt.subplot(1, 2, 2)
plt.plot(model_history["acc"])
plt.plot(model_history["val_acc"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model accuracy")
plt.legend(["train", "valid"])
plt.show()

# Augment test Dataset
test_generator = dataGenerator.flow_from_directory(test_dir,
                                                   target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                   classes=None,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   seed=123)

test_generator.reset()

# Test and show the result
pred = model.predict(test_generator, steps=len(test_generator), verbose=1)

pred_class_indices = np.argmax(pred, axis=1)

f, ax =plt.subplots(5, 5, figsize=(15, 15))

for i in range(25):
    bgr_img = cv2.imread(test_dir+test_generator.filenames[i])
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    predicted_class = "Dog" if pred_class_indices[i] else "Cat"

    ax[i//5, i%5].imshow(rgb_img)
    ax[i // 5, i % 5].axis("off")
    ax[i // 5, i % 5].set_title("Predicted:{}".format(predicted_class))
plt.show()

# Check main condition
"""Not needed"""

