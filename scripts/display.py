import tensorflow as tf
import matplotlib.pyplot as plt
import os

batch_size = 32
dim = 128
if os.name == 'nt':
    prefix = 'D:/'
if os.name == 'posix':
    prefix = '/media/bob/WOLAND/'
if os.name == 'posix':
    data_dir = '/home/bob/fungi-id-ai/images_' + str(dim)
if os.name == 'nt':
    data_dir = prefix + '/PROJLIB/Python/fungi-id-ai/images_' + str(dim)

with tf.device("/cpu:0"):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(dim, dim),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(dim, dim),
        batch_size=batch_size)

    class_names = train_ds.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")