import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open the HDF5 dataset file
h5f = h5py.File('D:/PROJLIB/Python/fungi_id/model/fungi_model_rgb_128.h5', 'r')

# Load the data from the HDF5 file
images = h5f['images'][:]
labels = h5f['labels'][:]

# Close the HDF5 file
h5f.close()

# Plot the first 5 images from the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i])
    plt.xlabel('Label: {}'.format(labels[i]))

plt.show()

