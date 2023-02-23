import cv2
import os
import glob
import exifread
prefix = '/media/bob/WOLAND/'
if os.name == 'nt':
    prefix = 'D:/'
if os.name == 'posix':
    prefix = '/media/bob/WOLAND/'

# Directory containing the raw images
raw_images_dir = prefix + 'Fungarium.backup'
# Directory to store the processed images
dim = 256
if os.name == 'posix':
    processed_images_dir = '/home/bob/fungi-id-ai/images_30' + str(dim)
if os.name == 'nt':
    processed_images_dir = prefix + '/PROJLIB/Python/fungi-id-ai/images_' + str(dim)

# Helper function to get date picture taken from image file
def get_date_picture_taken(filename):
    with open(filename, 'rb') as f:
        tags = exifread.process_file(f)
        if 'EXIF DateTimeOriginal' in tags:
            return tags['EXIF DateTimeOriginal'].values.replace(":","-")
    return None

# Load the raw images
def process_directory(dir):
    counter = len(glob.glob1(dir,'*.jpg'))
    if counter < 30: 
        return
    images = []
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isfile(file_path) and (filename.endswith(".jpg") or filename.endswith(".jpeg")):
            print(f"Adding image {filename}")
            image = cv2.imread(file_path)
            images.append(image)

    # Resize the images to dimxdim
    processed_images = []
    for i, image in enumerate(images):
        print(f"Processing image {i+1} of {len(images)}")

        # Crop image
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
        # NOSONAR processed_image = cv2.resize(cropped_image, (dim, dim), interpolation=cv2.INTER_AREA)
        # resized_image = tf.image.resize(tf.expand_dims(cropped_image, 0), (dim, dim), method='area')
        # processed_image = resized_image.numpy().squeeze()
        
        processed_image = cv2.resize(cropped_image, (dim, dim), interpolation=cv2.INTER_AREA)
        processed_images.append(processed_image)

    # Save the processed images to disk
    processed_images_dir_new = os.path.join(processed_images_dir, os.path.basename(os.path.normpath(dir)))
    if not os.path.exists(processed_images_dir_new):
        os.makedirs(processed_images_dir_new)
        
    for i, (processed_image, filename) in enumerate(zip(processed_images, os.listdir(dir))):
        folder_name = os.path.basename(dir)
        date_picture_taken = get_date_picture_taken(os.path.join(dir, filename))
        processed_image_filename = '{}_{}.jpg'.format(folder_name, i)
        if date_picture_taken:
            processed_image_filename = '{}_{}_{}.jpg'.format(folder_name, date_picture_taken, i) 
        print(f"Saving image {i+1} of {len(processed_images)} - {processed_image_filename}")
        processed_image_filepath = os.path.join(processed_images_dir_new, processed_image_filename)
        cv2.imwrite(processed_image_filepath, processed_image)

# Recursively process the raw images directory
for root, dirs, files in os.walk(raw_images_dir):
    process_directory(root)