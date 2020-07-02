# Imports
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import cifar10
from keras.utils import to_categorical
import os
from matplotlib import pyplot as plt

# Hyper parameters
IMAGE_SIZE = 32 # cifar10 size
NUM_CHANNELS = 3
NUM_CLASSES = 10

TRAIN_BS = 64
VALID_BS = 64

EARLY_STOPPING_PATIENCE = 5
valid_steps = 20

lr = 0.01
moment = 0.9
decay = 1e-6
epochs = 10

train_dir = "./train"
valid_dir = "./valid"
test_dir = "./test"

weights_name = "best_model.h5"

#Load the dataset

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Model construction
model = Sequential()

base_model = ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
                      include_top=False,
                      pooling = "avg")

model.add(base_model)
model.add(Dense(NUM_CLASSES, activation = "softmax"))

# model.layers[0].trainable = False # Forgot to write in the first place !!!!

model.summary()

# Model Compilation
# opt = SGD(lr = lr,
#           momentum = moment,
#           decay = decay,
#           nesterov = True)
opt = Adam()
model.compile(opt,
              loss="categorical_crossentropy",
              metrics=["acc"])

# Augment training and validation Dataset
dataGenerator = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = dataGenerator.flow(x_train,
                                     y_train,
                                     batch_size=TRAIN_BS)


valid_generator = dataGenerator.flow(x_test,
                                     y_test,
                                     batch_size=VALID_BS)

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
                                        steps_per_epoch=x_train.shape[0]//TRAIN_BS,
                                        epochs=epochs,
                                        verbose=1,
                                        callbacks=[checkpoint, earlyStop],
                                        validation_data=valid_generator,
                                        validation_steps=valid_steps)

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

# Test and show the result

# Check main condition

