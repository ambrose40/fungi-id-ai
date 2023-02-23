import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

args = sys.argv[1:]
dim = int(args[0]) #128
output_filename = args[1] #'fungi_pretrained_model'
batch_size = int(args[2]) #32

if dim == '':
    dim = 128
if output_filename == '':
    output_filename = 'fungi_pretrained_model'
if batch_size == '':
    batch_size = 32

if os.name == 'nt':
    prefix = 'D:/'
if os.name == 'posix':
    prefix = '/media/bob/WOLAND/'
if os.name == 'posix':
    data_dir = '/home/bob/fungi-id-ai/images_' + str(dim)
if os.name == 'nt':
    data_dir = prefix + '/PROJLIB/Python/fungi-id-ai/images_' + str(dim)

with tf.device("/gpu:0"):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=6666,
        image_size=(dim, dim),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=6666,
        image_size=(dim, dim),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    print(num_classes)

    in_shape = (dim, dim, 3)
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=in_shape)

    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),
                                          tf.keras.layers.RandomRotation(0.3),
                                          tf.keras.layers.RandomZoom(0.2)])

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10,
                                               monitor='val_accuracy',
                                               restore_best_weights=True)]

    # data_augmentation = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.RandomFlip("horizontal_and_vertical",
    #                         input_shape=(dim, dim, 3)),
    #         tf.keras.layers.RandomRotation(0.22),
    #         tf.keras.layers.RandomZoom(0.22),
    #         # tf.keras.layers.RandomWidth(0.13),
    #         # tf.keras.layers.RandomHeight(0.13),
    #         tf.keras.layers.RandomContrast(factor=0.22), 
    #         tf.keras.layers.RandomBrightness(factor=0.22)
    #     ]
    # )

    model = tf.keras.models.Sequential()
    model.add(data_augmentation)
    
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(dim, dim, 3)))
    model.add(tf.keras.layers.Dense(1024, activation='relu', input_dim=512))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(dim, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # model = tf.keras.Sequential([
    #     data_augmentation,
    #     tf.keras.layers.Rescaling(1./255, input_shape=(dim, dim, 3)),
    #     tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(96, 3, padding='same', activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Dropout(0.13),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(dim, activation='relu'),
    #     tf.keras.layers.Dense(num_classes, name="outputs")
    # ])

    # Compile the model

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=300,
        callbacks=callbacks
    )
    
    if os.name == 'posix':
        path = '/home/bob/fungi-id-ai/model/'
    if os.name == 'nt':
        path = prefix + '/PROJLIB/Python/fungi-id-ai/model/'
    model.save_weights(path + output_filename  + '_' + str(dim) + '_weights.ckpt')

    model.save(path + output_filename + '_' + str(dim) + '.h5')
    model.summary()