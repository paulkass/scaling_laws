import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#dataset = tf.keras.datasets.cifar10
y_binary_train = tf.keras.utils.to_categorical(y_train)
y_binary_test = tf.keras.utils.to_categorical(y_test)

output_size = 10

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(output_size, activation='relu'))
model.add(tf.keras.layers.Softmax())

model.compile(optimizer = tf.keras.optimizers.RMSprop(),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"])

model.fit(x_train,
        y_binary_train,
        epochs = 20,
        validation_data = (x_test, y_binary_test)
)

