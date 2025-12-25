import tensorflow as tf
from src.dataset import build_dataset
from src.model import build_model


BATCH_SIZE = 32
EPOCHS = 15

image_dir = "./data/images"
xml_dir = "./data/annotations"

dataset = build_dataset(image_dir, xml_dir, BATCH_SIZE)

total = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.7 * total)
val_size = int(0.15 * total)

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size).take(val_size)

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        "bbox": "mse",
        "class": "categorical_crossentropy"
    },
    metrics={"class": "accuracy"}
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("./saved_model/mask_detector.keras")
