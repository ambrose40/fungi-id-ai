import cv2
import h5py
import numpy as np
import os
import glob
import pickle

# Create lists to store all the image arrays and labels
images_list = []
labels_list = []

texts = []

# Set to store the processed filenames
processed_filenames = set()
processed_folders = set()
# Directory containing the raw images
images_dir = '/home/bob/fungi-id-ai/images_gpu_128'
# Directory to store the processed images
processed_dir = '/home/bob/fungi-id-ai/model/fungi_model_filtered_40.h5'

# write list to binary file
def write_list(a_list):
    # store list in binary file so 'wb' mode
    with open('/home/bob/fungi-id-ai/model/fungi_model_texts_filtered_40.bin', 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')

# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('/home/bob/fungi-id-ai/model/fungi_model_texts_filtered_40.bin', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def process_directory(dir):
    folder_name = os.path.basename(dir)
    # Loop through all the files in the folder
    counter = len(glob.glob1(dir,'*.jpg'))
    if counter < 40: 
        return
    for filename in os.listdir(dir):
        # Check if the file is a jpeg image
        if filename.endswith('.jpg'):
            label = filename.split("_")[0]
            file_path = os.path.join(dir, filename)
            # Check if the file has already been processed
            if file_path in processed_filenames:
                continue
            processed_filenames.add(file_path)
            if folder_name not in processed_folders:
                processed_folders.add(folder_name)
                texts.append(label)
            label_id = len(processed_folders) - 1
            try:
                # Load the image using OpenCV
                img = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Preprocess the image using OpenCV functions
                # Replace this section with your own preprocessing steps
                # No need to resize, all my images are already of 384x384 
                # (the size I want to use for fungi AI object model)
                # img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                # img = img.astype('float32') / 255.0
                
                # Add the processed image to the list
                images_list.append(img)
                
                # Add the label for the image
                # Replace 0 with the actual label for the image
                
            
                labels_list.append(label_id)

                print(f"Processed image: {filename}, Label: {label}, Id: {label_id}")
                
            except Exception as e:
                print(f"Error processing image: {filename}. Error: {e}")
    
# Recursively process the raw images directory
total_files = 0
processed_files = 0

for root, dirs, files in os.walk(images_dir):
    total_files = len(files)
    process_directory(root)
    processed_files += 1

# Convert the lists of images and labels to NumPy arrays
images = np.array(images_list)
labels = np.array(labels_list)
# Use LabelEncoder to convert the labels to integers
# le = LabelEncoder()
# labels = le.fit_transform(labels_list)
try:
    # Save the processed images and labels to a HDF5 file
    with h5py.File(processed_dir, 'w') as f:
        f.create_dataset('images', data=images)
        f.create_dataset('labels', data=labels)
except Exception as e:
    print(f"Error saving processed images and labels to file. Error: {e}")

try:
    write_list(texts)
except Exception as e:
    print(f"Error saving processed texts to file. Error: {e}")

print("\nProcessing complete. Results:")
print(f"Total files in images directory: {total_files}")
print(f"Processed files: {processed_files}")
print(f"Failed to process files: {total_files - processed_files}")