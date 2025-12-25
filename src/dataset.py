import tensorflow as tf
import xml.etree.ElementTree as ET
import os

IMG_SIZE = 224
NUM_CLASSES = 3

CLASSES = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    boxes, labels = [], []

    for obj in root.findall("object"):
        label = CLASSES[obj.find("name").text]
        box = obj.find("bndbox")

        xmin = int(box.find("xmin").text) / width
        ymin = int(box.find("ymin").text) / height
        xmax = int(box.find("xmax").text) / width
        ymax = int(box.find("ymax").text) / height

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    # choose largest face
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = areas.index(max(areas))

    return boxes[idx], labels[idx]

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

def tf_wrapper(img_path, xml_path):
    def _parse(img_p, xml_p):
        # img_p is tf.string → tf.io.read_file accepts it directly
        img = load_image(img_p)

        # xml parser needs Python string
        box, label = parse_xml(xml_p.numpy().decode("utf-8"))

        return img, box, tf.one_hot(label, NUM_CLASSES)

    img, box, label = tf.py_function(
        _parse,
        [img_path, xml_path],
        [tf.float32, tf.float32, tf.float32]
    )

    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    box.set_shape((4,))
    label.set_shape((NUM_CLASSES,))

    return img, {"bbox": box, "class": label}

def build_dataset(image_dir, xml_dir, batch_size=32):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    xml_paths = sorted([os.path.join(xml_dir, f) for f in os.listdir(xml_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, xml_paths))
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
