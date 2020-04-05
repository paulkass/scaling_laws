import numpy as np
import tensorflow as tf
import sys
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

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
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(output_size, activation='softmax'))

model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = ["accuracy"])

model.fit(x_train,
        y_train,
        epochs = 7,
        verbose = 1,
        validation_split = 0.3
)

print(model.evaluate(x_test, y_test, verbose = 1))
