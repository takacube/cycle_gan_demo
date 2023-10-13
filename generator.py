from __future__ import print_function, division
from keras.layers import Input
from keras.layers import UpSampling2D, Conv2D
from keras.models import Model

class Generator():
    @staticmethod
    def build_generator(img_shape, gf, conv2d, deconv2d, channels):
        """U-Net Generator"""
        # Image input
        d0 = Input(shape=img_shape)

        # Downsampling
        d1 = conv2d(d0, gf)
        d2 = conv2d(d1, gf * 2)
        d3 = conv2d(d2, gf * 4)
        d4 = conv2d(d3, gf * 8)

        # Upsampling
        u1 = deconv2d(d4, d3, gf * 4)
        u2 = deconv2d(u1, d2, gf * 2)
        u3 = deconv2d(u2, d1, gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(channels, kernel_size=4,
                            strides=1, padding='same', activation='tanh')(u4)

        model = Model(d0, output_img)
        return model