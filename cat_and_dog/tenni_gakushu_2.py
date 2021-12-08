from matplotlib.colors import from_levels_and_colors
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import keras

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.ops.control_flow_ops import from_control_flow_context_def
from tensorflow.python.ops.gen_array_ops import tensor_scatter_add_eager_fallback, tensor_strided_slice_update
from tensorflow.python.ops.gen_math_ops import less_eager_fallback
from tensorflow.python.training.tracking import base

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file("cats_and_dogs.zip",origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

class_names = train_dataset.class_names

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print("Number of Validation batches: %d" % tf.data.experimental.cardinality(validation_dataset))
print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

def tenni_gakushu(base):
        
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

    IMG_SHAPE = IMG_SIZE + (3,)
    base_model_1 = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_3 = tf.keras.applications.MobileNetV3Small(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_4 = tf.keras.applications.xception.Xception(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_5 = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_6 = tf.keras.applications.vgg19.VGG19(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_7 = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_8 = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_9 = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_10 = tf.keras.applications.densenet.DenseNet121(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_11 = tf.keras.applications.densenet.DenseNet169(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    base_model_12 = tf.keras.applications.densenet.DenseNet201(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    #base_model_13 = tf.keras.applications.nasnet.NASNetLarge(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    #base_model_14 = tf.keras.applications.nasnet.NASNetMobile(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

    if base == 1:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v1"
        base_model = base_model_1
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
    
    elif base == 2:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v2"
        base_model = base_model_2
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    elif base == 3:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v3"
        base_model = base_model_3
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    elif base == 4:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v4"
        base_model = base_model_4
        preprocess_input = tf.keras.applications.xception.preprocess_input 
    
    elif base == 5:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v5"
        base_model = base_model_5
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
    
    elif base == 6:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v6"
        base_model = base_model_6
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
    
    elif base == 7:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v7"
        base_model = base_model_7
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    
    elif base == 8:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v8"
        base_model = base_model_8
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    
    elif base == 9:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v9"
        base_model = base_model_9
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
    
    elif base == 10:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v10"
        base_model = base_model_10
        preprocess_input = tf.keras.applications.densenet.preprocess_input
    
    elif base == 11:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v11"
        base_model = base_model_11
        preprocess_input = tf.keras.applications.densenet.preprocess_input 
    
    elif base == 12:
        log_dir = "C:/Users/nf_maehata/Desktop/転移学習/tensorboard/v12"
        base_model = base_model_12
        preprocess_input = tf.keras.applications.densenet.preprocess_input 


    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False
    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(160,160,3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs,outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])

    model.summary()

    len(model.trainable_variables)

    initial_epochs = 10
    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}" .format(loss0))
    print("initial accuracy: {:.2f}" .format(accuracy0))

    history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset, callbacks=tensorboard_callback)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim(0.8,1.0)
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    base_model.trainable = True

    print("Number of layers in the bass model: ", len(base_model.layers))

    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10), metrics=["accuracy"])
    model.summary()

    len(model.trainable_variables)

    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs
    history_fine = model.fit(
        train_dataset,epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
        callbacks=tensorboard_callback
        )

    acc += history_fine.history["accuracy"]
    val_acc += history_fine.history["val_accuracy"]

    loss += history_fine.history["loss"]
    val_loss += history_fine.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    loss, accuracy = model.evaluate(test_dataset)
    print("Test accuracy :", accuracy)

    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].astype("uint8"))
        plt.title(class_names[predictions[i]])
        plt.axis("off")

tenni_gakushu(1)
tenni_gakushu(2)
tenni_gakushu(3)
tenni_gakushu(4)
tenni_gakushu(5)
tenni_gakushu(6)
tenni_gakushu(7)
tenni_gakushu(8)
tenni_gakushu(9)
tenni_gakushu(10)
tenni_gakushu(11)
tenni_gakushu(12)
