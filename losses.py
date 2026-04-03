#losses
import tensorflow as tf


class Losses:
    def __init__(self):
        # from_logits=True because discriminator outputs raw logits (no sigmoid)
        self.adversarial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss(self, disc_generated, generated_output, target_image):
        """
        Generator loss = GAN loss + 100 * L1 loss
        - GAN loss: fool the discriminator (generated patches should look real)
        - L1 loss:  pixel-wise reconstruction to keep outputs sharp and accurate
        Weight of 100 on L1 matches the original Pix2Pix paper.
        """
        gan_loss = self.adversarial_loss(tf.ones_like(disc_generated), disc_generated)
        l1_loss  = tf.reduce_mean(tf.abs(target_image - generated_output))
        total_loss = gan_loss + (100 * l1_loss)
        return total_loss, gan_loss, l1_loss

    def discriminator_loss(self, real_output, generated_output):
        """
        Discriminator loss = real loss + fake loss
        - real_loss: real pairs should be classified as real (label = 1)
        - fake_loss: generated pairs should be classified as fake (label = 0)
        """
        real_loss = self.adversarial_loss(tf.ones_like(real_output),  real_output)
        fake_loss = self.adversarial_loss(tf.zeros_like(generated_output), generated_output)
        return real_loss + fake_loss