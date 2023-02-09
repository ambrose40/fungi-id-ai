import tensorflow as tf
import numpy as np
import sys


with tf.device("/gpu:0"):
    dim = 128
    data_dir = '/home/bob/fungi-id-ai/images_128/'
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

    model = tf.keras.models.load_model('/home/bob/fungi-id-ai/model/fungi_pretrained_model_filtered_20.h5')

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
