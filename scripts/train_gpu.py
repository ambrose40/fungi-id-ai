import tensorflow as tf
import matplotlib.pyplot as plt
import os

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

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

with tf.device("/gpu:0"):
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
    print(class_names)


    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal",
                            input_shape=(dim, dim, 3)),
            tf.keras.layers.RandomRotation(0.18),
            tf.keras.layers.RandomZoom(0.18),
        ]
    )

    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, name="outputs")
    ])

    # Compile the model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_classes
    )

    if os.name == 'posix':
        path = '/home/bob/fungi-id-ai/model/fungi_pretrained_model_' + str(dim) + 'h5'
    if os.name == 'nt':
        path = prefix + '/PROJLIB/Python/fungi-id-ai/model/fungi_pretrained_model_' + str(dim) + 'h5'

    model.save(path)
    model.summary()