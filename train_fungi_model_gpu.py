import tensorflow as tf
import h5py

with tf.device("/gpu:0"):
    # Open the HDF5 dataset file
    h5f = h5py.File('D:/PROJLIB/Python/fungi_id/model/fungi_model_rgb_128_full.h5', 'r')

    # Load the data from the HDF5 file
    x_train = h5f['images'][:]

    y_train = tf.keras.utils.to_categorical(h5f['labels'][:], 497)
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'), # upscale 32 x 64 x 128 -> 384 x 768 x 1024
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(497, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=497, batch_size=128, validation_split=0)
    model.save('D:/PROJLIB/Python/fungi_id/model/fungi_pretrained_model_rgb_128_full.h5')

    # Close the HDF5 file
    h5f.close()