#discriminator
import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, concatenate
from keras.initializers import RandomNormal


class Discriminator:
    def __init__(self):

        self.initializer = RandomNormal(stddev=0.02, seed=42)

    def downscale(self, filters, batch_norm=True):
        """
        PatchGAN discriminator block: Conv -> BN (optional) -> LeakyReLU
        The first layer skips BN as per the original Pix2Pix spec.
        """
        layers = [
            Conv2D(
                filters,
                kernel_size=4,
                strides=2,
                padding='same',
                kernel_initializer=self.initializer,
                use_bias=False
            ),
        ]
        if batch_norm:
            layers.append(BatchNormalization())
        layers.append(LeakyReLU(alpha=0.2))
        return tf.keras.Sequential(layers)

    def build_discriminator(self):
        image  = Input(shape=(128, 128, 3), name="ImageInput")
        target = Input(shape=(128, 128, 3), name="TargetInput")

        x = concatenate([image, target])  

        # Full 4-stage PatchGAN: C64 -> C128 -> C256 -> C512
        x = self.downscale(64,  batch_norm=False)(x)   # (128, 128, 64)  — no BN on first layer
        x = self.downscale(128)(x)                     # (64,  64,  128)
        x = self.downscale(256)(x)                     # (32,  32,  256)
        x = self.downscale(512)(x)                     # (16,  16,  512)

        # Stride-1 conv to increase depth before final patch prediction
        x = Conv2D(512, kernel_size=4, strides=1, padding='same',
                   kernel_initializer=self.initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Final patch output — no activation (logits for BCEWithLogits)
        x = Conv2D(1, kernel_size=4, strides=1, padding='same',
                   kernel_initializer=self.initializer)(x)

        return Model(inputs=[image, target], outputs=x, name="Discriminator")