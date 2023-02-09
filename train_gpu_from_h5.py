import tensorflow as tf
import h5py

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

count = 104 #497
dim = 128
batch_size = 32
with tf.device("/gpu:0"):
    # Open the HDF5 dataset file
    h5f = h5py.File('/home/bob/fungi-id-ai/model/fungi_model_filtered_40.h5', 'r')

    # Load the data from the HDF5 file
    x_train = h5f['images'][:]

    y_train = tf.keras.utils.to_categorical(h5f['labels'][:], count)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal",
                            input_shape=(dim, dim, 3)),
            tf.keras.layers.RandomRotation(0.4444),
            tf.keras.layers.RandomZoom(0.4444),
        ]
    )

    # Define the model architecture
    model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Conv2D(dim, (3, 3), activation='relu', input_shape=(dim, dim, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(dim*2, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(dim*4, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dim*4, activation='relu'),
        tf.keras.layers.Dense(count, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=count, batch_size=batch_size, validation_split=0.2)
    model.save('/home/bob/fungi-id-ai/model/fungi_pretrained_model_filtered_40.h5')

    # Close the HDF5 file
    h5f.close()