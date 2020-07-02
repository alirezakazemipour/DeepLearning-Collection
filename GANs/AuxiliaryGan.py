# Imports
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization
from keras.layers import Activation, LeakyReLU, Conv2D, Flatten, Input, concatenate
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
import os


'''Dynamic and Hyper parameters'''
latent_dim = 100
image_size = 28
image_shape = (image_size, image_size, 1)

gen_filters = [128, 64, 32, 1]
gen_strides = [2, 2, 1, 1]

dis_filters = [32, 64, 128, 256]
dis_strides = [2, 2, 2, 1]

kernel_size = (5, 5)
alpha = 0.2

dis_lr = 2e-4
dis_decay = 6e-8
dis_opt = RMSprop(lr = dis_lr, decay = dis_decay)

adv_lr = dis_lr * 0.5
adv_decay = dis_decay * 0.5
adv_opt = RMSprop(lr = adv_lr, decay = adv_decay)

test_size = 16
batch_size = 64
epochs = 40000

save_intervals = 500
save_adress = "./images"

'''Download dataset'''
(x_train, y_train), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(-1, image_size, image_size, 1).astype("float32") / 255
y_train = to_categorical(y_train)

'''BatchNormalization-Relu is used repeatedly so get it into a function'''
def BN_RELU(x):

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

'''Build Generator'''
def build_generator(z_inputs, label_inputs, image_size = 28):

    image_size = image_size // 4

    x = concatenate([z_inputs, label_inputs])
    x = Dense(image_size * image_size * gen_filters[0])(x)
    x = Reshape((image_size, image_size, gen_filters[0]))(x)

    for stride, filter in zip(gen_strides, gen_filters):
        x = BN_RELU(x)
        x = Conv2DTranspose(filters = filter,
                            strides = stride,
                            padding = "same",
                            kernel_size = kernel_size)(x)
    outputs = Activation("sigmoid")(x)

    model = Model([z_inputs, label_inputs], outputs, name = "Generator")
    model.summary()
    return model

"""Build Discriminator"""
def build_discriminator(inputs):

    x = inputs

    for stride, filter in zip(dis_strides, dis_filters):
        x = LeakyReLU(alpha = alpha)(x)
        x = Conv2D(filters = filter,
                   strides = stride,
                   padding = "same",
                   kernel_size = kernel_size)(x)
    y = Flatten()(x)
    x = Dense(1)(y)
    rf_outputs = Activation("sigmoid")(x)

    x = Dense(128)(y)
    x = Dense(10)(x)
    label_outputs = Activation("softmax")(x)

    model = Model(inputs, [rf_outputs, label_outputs], name = "Discriminator")
    model.summary()
    return model

"""Construct models """
def build_and_train():

    dis_inputs = Input(shape = image_shape)
    dis = build_discriminator(dis_inputs)
    dis.compile(optimizer = dis_opt,
                loss = ["binary_crossentropy", "categorical_crossentropy"],
                metrics = ["acc"])

    gen_z_inputs = Input(shape = (latent_dim,))
    gen_label_inputs = Input( shape=(10,) )
    gen = build_generator(z_inputs=gen_z_inputs, label_inputs=gen_label_inputs)

    dis.trainable = False
    adv_inputs = [gen_z_inputs, gen_label_inputs]
    adv_outputs = dis(gen(adv_inputs))
    adv = Model(adv_inputs, adv_outputs, name="Adversarial")
    adv.summary()
    adv.compile(optimizer = adv_opt,
                loss = ["binary_crossentropy", "categorical_crossentropy"],
                metrics = ["acc"])

    models = gen, dis, adv
    train(models)

"""Training phase"""
def train(models):

    gen, dis, adv = models
    test_noise = np.random.uniform(low = -1, high = 1, size =(test_size, latent_dim))
    test_labels = np.eye(10)[np.arange(batch_size) % 10]

    for epoch in range(1, epochs + 1): # (+ 1) for running plot function
        random_indices = np.random.randint(low = 0, high = x_train.shape[0], size = batch_size)
        real_imgs = x_train[random_indices]
        real_rf = np.ones(shape = (batch_size, 1))
        real_labels = y_train[random_indices]

        z = np.random.uniform(low = -1, high = 1, size =(batch_size, latent_dim))
        fake_labels = np.eye(10)[np.arange(batch_size) % 10]
        fake_imgs = gen.predict([z, fake_labels])
        fake_rf = np.zeros(shape = (batch_size, 1))

        dis_x = np.concatenate([real_imgs, fake_imgs])
        dis_rf_y = np.concatenate([real_rf, fake_rf])
        dis_labels_y = np.concatenate([real_labels, fake_labels])

        loss, _, _, _, _ = dis.train_on_batch(dis_x, [dis_rf_y, dis_labels_y])

        log = "step:{} dis[loss:{:3f}]".format(epoch, loss)

        adv_z_x = np.random.uniform( low=-1, high=1, size=(batch_size, latent_dim) )
        adv_labels_x = np.eye(10)[np.arange(batch_size) % 10]
        adv_rf_y = np.ones(shape = (batch_size, 1))

        loss, _, _, _, _ = adv.train_on_batch([adv_z_x, adv_labels_x], [adv_rf_y, adv_labels_x])

        log += " adv[loss:{:3f}]".format( loss )

        print(log)

        if epoch % save_intervals == 0:
            plot_image(gen, test_noise,test_labels, epoch)


"""Show what is being produced"""
def plot_image(generator, input_noise,labels,  step, show = False):

    z = input_noise
    n_images = z.shape[0]
    images = generator.predict([z, labels])

    rows = np.sqrt(n_images)

    plt.figure(figsize=(2,2))

    for i in range(n_images):
        plt.subplot(rows, rows, i+1)
        plt.imshow(images[i].reshape(image_size, image_size), cmap="gray")
        plt.axis("off")
    plt.savefig(os.path.join(save_adress, "step{}.png".format(step)))

    if show:
        plt.show()
    else:
        plt.close("all")


"""Main"""
if __name__ == "__main__":

    build_and_train()