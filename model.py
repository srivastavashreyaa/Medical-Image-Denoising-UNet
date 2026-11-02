"""U-Net for grayscale image denoising (encoder 32-64-128-256, bottleneck 512)."""
from tensorflow.keras import Model, layers


def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Activation("relu")(x)
    return x


def build_unet(size=256, base=32, depth=4):
    inp = layers.Input((size, size, 1))
    x, skips = inp, []
    for d in range(depth):                       # encoder
        x = conv_block(x, base * 2 ** d)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = conv_block(x, base * 2 ** depth)          # bottleneck
    for d in reversed(range(depth)):              # decoder
        x = layers.Conv2DTranspose(base * 2 ** d, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skips[d]])   # skip connection
        x = conv_block(x, base * 2 ** d)
    out = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(x)
    return Model(inp, out, name="denoising_unet")


if __name__ == "__main__":
    build_unet().summary()
