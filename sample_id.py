import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)

# y_test = keras.utils.to_categorical(y_test, 10)
# x_test = x_test.astype('float32') / 255.0

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

model.save('D:/PROJLIB/Python/fungi_id/model/fungi_pretrained_model.h5')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=32)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)