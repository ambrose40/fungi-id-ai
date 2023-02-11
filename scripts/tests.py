import tensorflow as tf
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

with tf.device("/gpu:0"):
    dim = 256
    if os.name == 'nt':
        prefix = 'D:/'
    if os.name == 'posix':
        prefix = '/media/bob/WOLAND/'
    if os.name == 'posix':
        data_dir = '/home/bob/fungi-id-ai/images_' + str(dim)
    if os.name == 'nt':
        data_dir = prefix + '/PROJLIB/Python/fungi-id-ai/images_' + str(dim)
    print(sys.argv[1:])
    image_url = sys.argv[1:][0]
    print(image_url)
    image = tf.keras.utils.get_file(str(hash(image_url)), origin=str(image_url))

    img = tf.keras.utils.load_img(
        image, target_size=(dim, dim), keep_aspect_ratio=True
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0,
            image_size=(dim, dim),
            batch_size=256)

    class_names = train_ds.class_names
    labels = train_ds

    if os.name == 'posix':
        path = '/home/bob/fungi-id-ai/model/fungi_id_model_' + str(dim) + '.h5'
    if os.name == 'nt':
        path = prefix + '/PROJLIB/Python/fungi-id-ai/model/fungi_id_model_' + str(dim) + '.h5'
    
    model = tf.keras.models.load_model(path)

    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    scores_numpy = scores.numpy()
    print("This image most likely belongs to {} with a {:.2f} percent confidence. Id: {}".format(class_names[np.argmax(scores)], 100 * np.max(scores), np.argmax(scores)))
    print("This image most likely belongs to {} with a {:.2f} percent confidence. Id: {}".format(class_names[np.argsort(scores)[-2]], 100 * np.sort(scores)[-2], np.argsort(scores)[-2]))
    print("This image most likely belongs to {} with a {:.2f} percent confidence. Id: {}".format(class_names[np.argsort(scores)[-3]], 100 * np.sort(scores)[-3], np.argsort(scores)[-3]))
    print("This image most likely belongs to {} with a {:.2f} percent confidence. Id: {}".format(class_names[np.argsort(scores)[-4]], 100 * np.sort(scores)[-4], np.argsort(scores)[-4]))

    filter = set()
    fig1 = plt.figure(figsize=(10, 10))
    fig1.suptitle('This image most likely belongs to: ', fontsize=16)

    k = 0 
    for images, labels in train_ds:
        if k == 0:
            k = k + 1
            j = 0
            for i in range(256):
                title = class_names[labels[i]]
                if (title not in filter) and (title == class_names[np.argmax(scores)]):
                    ax = plt.subplot(2, 2, 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title("id: {}, {:.2f} %".format(title, 100 * np.max(scores)))
                    plt.axis("off")
                    filter.add(title)
                    j = j + 1
                if (title not in filter) and (title == class_names[np.argsort(scores)[-2]]):
                    ax = plt.subplot(2, 2, 2)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title("id: {}, {:.2f} %".format(title, 100 * np.sort(scores)[-2]))
                    plt.axis("off")
                    filter.add(title)
                    j = j + 1
                if (title not in filter) and (title == class_names[np.argsort(scores)[-3]]):
                    ax = plt.subplot(2, 2, 3)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title("id: {}, {:.2f} %".format(title, 100 * np.sort(scores)[-3]))
                    plt.axis("off")
                    filter.add(title)
                    j = j + 1
                if (title not in filter) and (title == class_names[np.argsort(scores)[-4]]):
                    ax = plt.subplot(2, 2, 4)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title("id: {}, {:.2f} %".format(title, 100 * np.sort(scores)[-4]))
                    plt.axis("off")
                    filter.add(title)
                    j = j + 1            

    fig2 = plt.figure(figsize=(5, 5))
    fig2.suptitle('Mushroom recognition by photo: ', fontsize=16)
    ax1 = plt.subplot(1, 1, 1)
    plt.imshow(img)
    plt.title("url: {}".format(str(image_url)))
    plt.axis("off")
    
    plt.show()