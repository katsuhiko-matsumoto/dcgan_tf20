
import tensorflow as tf
from tensorflow.keras import layers
import os
import time
import math

#deep64v2
class DCGAN_DEEP64v2(object):

    def __init__(self):
        #入力画像の大きさ
        self.input_width = 256
        self.input_height = 256
        #出力画像の大きさ
        self.output_width = 64
        self.output_height = 64
        self.c_dim = 1

    def gen_gene_and_disc(self):
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def get_input_width(self):
        return self.input_width

    def get_input_height(self):
        return self.input_height

    def get_output_width(self):
        return self.output_width

    def get_output_height(self):
        return self.output_height

    def get_cdim(self):
        return self.c_dim

    def set_cdim(self, cdim):
        self.c_dim = cdim

    def make_generator_model(self):
        model = tf.keras.Sequential()
        #model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.Dense(2*2*4096, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        #model.add(layers.Reshape((7, 7, 256)))
        #assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
        model.add(layers.Reshape((2, 2, 4096)))
        assert model.output_shape == (None, 2, 2, 4096)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(2048, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 2, 2, 2048)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1024, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 4, 4, 1024)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 512)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        #model.add(layers.Conv2DTranspose(c_dim, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        model.add(layers.Conv2DTranspose(self.c_dim, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.output_height, self.output_width, self.c_dim)

        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[self.output_height, self.output_width, self.c_dim]))
        print('##dis:',model.output_shape)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        print('##dis:',model.output_shape)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
        print('##dis:',model.output_shape)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same'))
        print('##dis:',model.output_shape)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(2048, (5, 5), strides=(2, 2), padding='same'))
        print('##dis:',model.output_shape)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        print('##dis:',model.output_shape)
        model.add(layers.Dense(1))
        print('##dis:',model.output_shape)

        return model
