import tensorflow as tf
import numpy as np
import sys
import os

with tf.device("/gpu:0"):
    dim = 128
    if os.name == 'nt':
        prefix = 'D:/'
    if os.name == 'posix':
        prefix = '/media/bob/WOLAND/'
    if os.name == 'posix':
        data_dir = '/home/bob/fungi-id-ai/images_' + str(dim)
    if os.name == 'nt':
        data_dir = prefix + '/PROJLIB/Python/fungi-id-ai/images_' + str(dim)

    image_url = sys.argv[1:]
    image_path = tf.keras.utils.get_file(origin=image_url)

    img = tf.keras.utils.load_img(
        image_path, target_size=(dim, dim)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(dim, dim),
            batch_size=32)

    class_names = train_ds.class_names

    if os.name == 'posix':
        path = '/home/bob/fungi-id-ai/model/fungi_pretrained_model_' + str(dim) + 'h5'
    if os.name == 'nt':
        path = prefix + '/PROJLIB/Python/fungi-id-ai/model/fungi_pretrained_model_' + str(dim) + 'h5'
    
    model = tf.keras.models.load_model(path)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
