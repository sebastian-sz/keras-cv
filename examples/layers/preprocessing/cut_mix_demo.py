"""cut_mix_demo.py shows how to use the CutMix preprocessing layer to 
preprocess the oxford_flowers102 dataset.  In this script the flowers 
are loaded, then are passed through the preprocessing layers.  
Finally, they are shown using matplotlib.
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import keras_cv.layers.preprocessing.cut_mix as cut_mix
import matplotlib.pyplot as plt


IMG_SIZE = (224, 224)
BATCH = 16


def resize(image, label, num_classes=10):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    label = tf.one_hot(label, num_classes)
    return image, label


def main():
    data, ds_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
    train_ds = data["train"]

    num_classes = ds_info.features["label"].num_classes

    train_ds = (
        train_ds.map(lambda x, y: resize(x, y, num_classes=num_classes))
        .shuffle(10 * BATCH)
        .batch(BATCH)
    )
    cutmix = cut_mix.CutMix()
    train_ds = train_ds.map(cutmix, num_parallel_calls=tf.data.AUTOTUNE)

    for images, labels in train_ds.take(1):
        plt.figure(figsize=(8, 8))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()