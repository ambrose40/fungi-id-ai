import tensorflow as tf
import h5py

# Check if GPU is available
if tf.test.is_gpu_available():
    # Use GPU
    with tf.device('GPU:0'):
        # Open the HDF5 dataset file
        h5f = h5py.File('D:/PROJLIB/Python/fungi_id/model/fungi_model.h5', 'r')

        # Load the data from the HDF5 file
        x_train = h5f['images'][:]
        x_train = x_train.astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(h5f['labels'][:], 497)

        # Define the model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(384, (3, 3), activation='relu', input_shape=(384, 384, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(497, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=497, batch_size=32, validation_split=0.2)
        model.save('D:/PROJLIB/Python/fungi_id/model/fungi_pretrained_model.h5')
        
        # Close the HDF5 file
        h5f.close()
else:
    # Open the HDF5 dataset file
    h5f = h5py.File('D:/PROJLIB/Python/fungi_id/model/fungi_model.h5', 'r')

    # Load the data from the HDF5 file
    x_train = h5f['images'][:]
    x_train = x_train.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(h5f['labels'][:], 497)

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(384, (3, 3), activation='relu', input_shape=(384, 384, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(497, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=497, batch_size=32, validation_split=0.2)
    model.save('D:/PROJLIB/Python/fungi_id/model/fungi_pretrained_model.h5')

    # Close the HDF5 file
    h5f.close()