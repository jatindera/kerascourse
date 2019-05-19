import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#plt.imshow(train_images[12])
#print(train_labels[12])
#print(train_images[12])
train_images = train_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
print('Evaluation result starts...')
model.evaluate(test_images,test_labels)
print('Evavluation result end...')
# Testing one image from test datasets
plt.imshow(test_images[2])
img = np.reshape(test_images[2],(1,784))
print(model.predict(img))

#img = io.imread('shirt2.png', as_gray=True)

#nx, ny = img.shape
#img_flat = img.reshape(nx * ny)
#img = np.reshape(img,(1,784))
#print(model.predict(img))
plt.show()
