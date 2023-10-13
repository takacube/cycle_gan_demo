from __future__ import print_function, division
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_loader import DataLoader
from discriminator import Discriminator
from generator import Generator
from layer_generator import conv2D, deconv2D
class CycleGAN():
    def __init__(self, img_rows=128, img_cols=128, channels=3):
        img_shape = (img_rows, img_cols, channels)
        patch = int(img_rows / 2**4)
        discriminator_patch = (patch, patch, 1)
        generator_filter_num = 32
        discriminator_filter_num = 64
        lambda_cycle = 10.0
        lambda_id = 0.9 * lambda_cycle
        descriminator = Discriminator()
        d_A = descriminator.build_discriminator(img_shape=img_shape, filter_num=discriminator_filter_num, conv2d=conv2D)
        d_B = descriminator.build_discriminator(img_shape=img_shape, filter_num=discriminator_filter_num, conv2d=conv2D)

        g_AB = Generator.build_generator(img_shape=img_shape, gf=generator_filter_num, conv2d=conv2D, deconv2d=deconv2D, channels=channels)
        g_BA = Generator.build_generator(img_shape=img_shape, gf=generator_filter_num, conv2d=conv2D, deconv2d=deconv2D, channels=channels)

        img_A = Input(shape=img_shape)
        img_B = Input(shape=img_shape)

        fake_B = g_AB(img_A)
        fake_A = g_BA(img_B)

        img_A_id = g_BA(img_A)
        img_B_id = g_AB(img_B)

        reconstr_A = g_BA(fake_B)
        reconstr_B = g_AB(fake_A)

        valid_A = d_A(fake_A)
        valid_B = d_B(fake_B)

        combined_model = Model(inputs=[img_A, img_B],
                                outputs=[valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id])
        combined_model.compile(loss=['mse', 'mse',
                                        'mae', 'mae',
                                        'mae', 'mae'],
                                loss_weights=[1, 1,
                                            lambda_cycle, lambda_cycle,
                                            lambda_id, lambda_id],
                                optimizer=tf.keras.optimizers.legacy.Adam(0.0002, 0.5)
                        )
        self.model = {"combined": combined_model, "g_AB": g_AB, "g_BA": g_BA, "d_A": d_A, "d_B": d_B, "discriminator_patch": discriminator_patch}

    def sample_images(self, current_epoch, current_batch_i):
        r, c = 2, 3
        dataset_name = 'apple2orange'
        data_loader = DataLoader(dataset_name=dataset_name, img_res=(128, 128))
        sample_imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        sample_imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        fake_B = self.model["g_AB"].predict(sample_imgs_A)
        fake_A = self.model["g_BA"].predict(sample_imgs_B)

        reconstr_A = self.model["g_BA"].predict(fake_B)
        reconstr_B = self.model["g_AB"].predict(fake_A)

        gen_imgs = np.concatenate([sample_imgs_A, fake_B, reconstr_A, sample_imgs_B, fake_A, reconstr_B])

        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("data_set/%s/%d_%d.png" % (dataset_name, current_epoch, current_batch_i))
        plt.show()

    def train(self, epochs, discriminator_patch, batch_size=1, sample_interval=50):
        valid = np.ones((batch_size,) + discriminator_patch)
        fake = np.zeros((batch_size,) + discriminator_patch)
        dataset_name = 'apple2orange'
        data_loader = DataLoader(dataset_name=dataset_name, img_res=(128, 128))

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):
                fake_B = self.model["g_AB"].predict(imgs_A)
                fake_A = self.model["g_BA"].predict(imgs_B)

                dA_loss_real = self.model["d_A"].train_on_batch(imgs_A, valid)
                dA_loss_fake = self.model["d_A"].train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.model["d_B"].train_on_batch(imgs_B, valid)
                dB_loss_fake = self.model["d_B"].train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                d_loss = 0.5 * np.add(dA_loss, dB_loss)
                g_loss = self.model["combined"].train_on_batch([imgs_A, imgs_B],
                                                    [valid, valid,
                                                    imgs_A, imgs_B,
                                                    imgs_A, imgs_B])
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
