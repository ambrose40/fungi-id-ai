import tensorflow as tf
import numpy as np
import h5py
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
else:
    print("No GPU found. Running on CPU.")

# Open the HDF5 dataset file
h5f = h5py.File('D:/PROJLIB/Python/fungi_id/model/fungi_model.h5', 'r')

# Load the data from the HDF5 file
model = tf.keras.models.load_model('D:/PROJLIB/Python/fungi_id/model/fungi_pretrained_model_gpu.h5')

# Load the image
image = cv2.imread('D:/PROJLIB/Python/fungi_id/data/image.jpg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Crop the image
h, w, c = image.shape
start_x = 0
start_y = 0
end_x = w
end_y = h
if w > h:
    start_x = (w - h) // 2
    end_x = w - start_x
if h > w:
    start_y = (h - w) // 2
    end_y = h - start_y

cropped_image = image[start_y:end_y, start_x:end_x]

# Resize the images to 128x128
processed_image = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_AREA)

# Preprocess the image using OpenCV functions
processed_image = processed_image.astype('float32') / 255.0

# Predict the class of the image
prediction = model.predict(np.array([processed_image]))

# Print the prediction
print(prediction[0])

predicted_class = np.argmax(prediction[0])

# Get the label of the predicted class
class_labels = h5f['labels'][:]
unique_classes = np.unique(class_labels)
predicted_label = unique_classes[predicted_class]

print("The predicted label for the new image is:", predicted_label)

# Calculate accuracy
accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(class_labels[predicted_class], class_labels[np.argmax(prediction[0])])
print("Accuracy: {:.2f}%".format(accuracy.result().numpy() * 100))

# The predicted label for the new image is: 131
# Close the HDF5 file
h5f.close()