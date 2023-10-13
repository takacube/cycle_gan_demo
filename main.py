from cycle_gan import CycleGAN
cycle_gan = CycleGAN()
cycle_gan.train(epochs=100, discriminator_patch=cycle_gan.model["discriminator_patch"])
cycle_gan