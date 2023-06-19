import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# plt.figure(figsize=(7, 2))
# plt.imshow(X_train[1])
# plt.show()

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

print(y_train[:5])


def plot_sample(X, y, index):
    plt.figure(figsize=(7, 2))
    plt.imshow(X[index])
    print(classes[y[index]])
    plt.xlabel(classes[y[index]])
    plt.show()


# plot_sample(X_train, y_train, 1)

X_train = X_train/255
X_test = X_test/255

cnn = keras.Sequential([
    # cnn
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)),
    keras.layers.MaxPool2D((2, 2)),

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPool2D((2, 2)),
    # dense
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

cnn.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(X_test, y_test)

y_pred = cnn.predict(X_test)
y_pred = [np.argmax(i) for i in y_pred]

print(y_pred[:5])
print(y_test[:5])
