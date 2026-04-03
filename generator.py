#generator.py
import tensorflow as tf
from keras import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, BatchNormalization, LeakyReLU, ReLU, Dropout, concatenate
from keras.initializers import RandomNormal
 
 
class Generator:
    def __init__(self):
        self.initializer = RandomNormal(stddev=0.02, seed=42)
 
    def downscale(self, filters):
        """Encoder block: Conv -> LeakyReLU -> BN"""
        return tf.keras.Sequential([
            Conv2D(
                filters,
                kernel_size=4,
                strides=2,
                padding='same',
                kernel_initializer=self.initializer,
                use_bias=False
            ),
            LeakyReLU(alpha=0.2),
            BatchNormalization()
        ])
 
    def upscale(self, filters, dropout=False):
        """
        Decoder block: ConvTranspose -> BN -> (optional Dropout 0.5) -> ReLU
        Dropout is applied to the first 3 decoder layers as per the Pix2Pix paper
        to introduce stochastic variation and prevent blurry outputs.
        FIX: Removed the redundant LeakyReLU that was before ReLU in the original.
        """
        layers = [
            Conv2DTranspose(
                filters,
                kernel_size=4,
                strides=2,
                padding='same',
                kernel_initializer=self.initializer,
                use_bias=False
            ),
            BatchNormalization(),
        ]
        if dropout:
            layers.append(Dropout(0.5))
        layers.append(ReLU())
        return tf.keras.Sequential(layers)
 
    def build_generator(self):
        inputs = Input(shape=(128, 128, 3), name="InputLayer")
 
        # Encoder: 7 downscale layers
        encoder = [self.downscale(f) for f in [64, 128, 256, 512, 512, 512]]
 
        # Bottleneck
        latent_space = self.downscale(512)
 
        # Decoder: first 3 layers have dropout=True (Pix2Pix paper spec)
        decoder_configs = [
            (512, True),
            (512, True),
            (512, False),
            (256, False),
            (128, False),
            (64,  False),
        ]
        decoder = [self.upscale(f, dropout=d) for f, d in decoder_configs]
 
        # Forward pass through encoder, collecting skip connections
        x = inputs
        skips = []
        for layer in encoder:
            x = layer(x)
            skips.append(x)
 
        # Bottleneck
        x = latent_space(x)
 
        # Forward pass through decoder with skip connections
        for up, skip in zip(decoder, reversed(skips)):
            x = up(x)
            x = concatenate([x, skip])
 
        # Final output layer: ConvTranspose -> tanh
        outputs = Conv2DTranspose(
            3,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=self.initializer,
            activation='tanh'
        )(x)
 
        return Model(inputs, outputs, name="Generator")
 