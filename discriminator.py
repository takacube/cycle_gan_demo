from __future__ import print_function, division
from keras.layers import Input
from keras.layers import Conv2D
from keras.models import Model
import tensorflow as tf

class Discriminator():
    def build_discriminator(self, img_shape, filter_num, conv2d):
        img = Input(shape=img_shape)

        d1 = conv2d(img, filter_num, normalization=False)
        d2 = conv2d(d1, filter_num * 2)
        d3 = conv2d(d2, filter_num * 4)
        d4 = conv2d(d3, filter_num * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        model = Model(img, validity)
        compiled_model = self._compile(model=model)
        disabled_model = self._disable_training(model=compiled_model)
        return disabled_model

    def _compile(self, model):
        optimizer = tf.keras.optimizers.legacy.Adam(0.0002, 0.5)
        model.compile(loss='mse',
                            optimizer=optimizer,
                            metrics=['accuracy'])
        return model

    def _disable_training(self, model):
        model.trainable = False
        return model