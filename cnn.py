import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfpl = tfp.layers

plt.rcParams['figure.figsize'] = (10, 6)
dataPath = './dataset/data'
target_shape = (512, 512)

train_datagen_nor = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_data = train_datagen_nor.flow_from_directory(
    dataPath + '/train',
    target_size=target_shape,
    color_mode='grayscale',
    subset='training',
    batch_size=32,
    class_mode='categorical'
)

valid_data = train_datagen_nor.flow_from_directory(
    dataPath + '/train',
    target_size=target_shape,
    color_mode='grayscale',
    subset='validation',
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)
input_shape = train_data.image_shape

def inspect_images(data, num_images):
    fig, ax = plt.subplots(nrows=1, ncols=num_images, figsize=(2 * num_images, 2))
    for i in range(num_images):
        ax[i].imshow(data[0][0][i, ..., 0], cmap='gray')
        ax[i].axis('off')
    plt.show()

inspect_images(data=train_data, num_images=8)

def get_deterministic_model(input_shape, loss, optimizer, metrics, num_classes):
    """
    This function should build and compile a CNN model according to the above specification.
    The function takes input_shape, loss, optimizer and metrics as arguments, which should be
    used to define and compile the model.
    Your function should return the compiled model.
    """
    model = Sequential([
        Conv2D(kernel_size=(5, 5), filters=8, activation='relu', padding='VALID', input_shape=input_shape),
        MaxPooling2D(pool_size=(6, 6)),
        Flatten(),
        Dense(units=num_classes, activation='softmax')
    ])

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

tf.random.set_seed(0)
deterministic_model = get_deterministic_model(
    input_shape=input_shape,
    loss=CategoricalCrossentropy(),
    optimizer=RMSprop(),
    metrics=['accuracy'],
    num_classes=num_classes
)
deterministic_model.summary()

tf.keras.utils.plot_model(deterministic_model, show_shapes=True)

import tensorflow.compat.v1 as tfs
gpu_options = tfs.GPUOptions(allow_growth=True)
session = tfs.InteractiveSession(config=tfs.ConfigProto(gpu_options=gpu_options))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

csv = CSVLogger("cnn-50-model_history_log(normal).csv", append=True)
es1 = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
mc = ModelCheckpoint('cnn-50-model.keras', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = deterministic_model.fit(
    train_data,
    epochs=300,
    shuffle=True,
    validation_data=valid_data,
    callbacks=[csv, es1, mc]
)

deterministic_model.save_weights('cnn_dataset-50_weights.weights.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.savefig('cnn_dataset-50-accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.savefig('cnn_dataset-50-loss.png')
plt.show()

train_acc = deterministic_model.evaluate(train_data)
val_acc = deterministic_model.evaluate(valid_data)
print("Train_Accuracy: %.2f%%" % (train_acc[1] * 100))
print("Valid_Accuracy: %.2f%%" % (val_acc[1] * 100))

test_datagen_nor = ImageDataGenerator(rescale=1./255)

test_generator_nor = test_datagen_nor.flow_from_directory(
    dataPath + '/test',
    target_size=(512, 512),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

test_acc = deterministic_model.evaluate(test_generator_nor)
print("Test_Accuracy: %.2f%%" % (test_acc[1] * 100))

# # Convert to TF Lite
# converter = tf.lite.TFLiteConverter.from_keras_model(deterministic_model)
# tflite_model = converter.convert()
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)

# # Example to import in TFLite
# import tflite_runtime.interpreter as tflite
# import numpy as np
# interpreter = tflite.Interpreter(model_path='model.tflite')

# print(interpreter.get_signature_list())
# cnn = interpreter.get_signature_runner()

# cnn.get_input_details()

# from matplotlib import image

# def load_image(path):
#     img = image.imread(path)
#     arr = np.array([img[:, :, 0]], dtype='float32')  # Assume grayscale
#     arr = np.resize(arr, (1, 512, 512, 1))
#     return arr

# cnn(conv2d_1_input=load_image(dataPath + '/rpi3_rot.png'))
