import tensorflow as tf

IMG_SIZE = 224
NUM_CLASSES = 3

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base.trainable = False

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    bbox = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)
    cls = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="class")(x)

    model = tf.keras.Model(inputs=base.input, outputs=[bbox, cls])
    return model
