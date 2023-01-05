import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(xtrain, ytrain),(xtest, ytest) = tf.keras.datasets.mnist.load_data()
xtrain, xtest = xtrain/255.0, xtest/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# predict
latest = tf.train.latest_checkpoint('my_mnist_checkpoint/ckpt')
model.load_weights(latest)
# predict the object
x = np.array(xtrain[:50])
y = model.predict(x)
print(y)

print(np.argmax(y,axis=1))
print(ytrain[:50])
# show
plt.imshow(xtrain[0],cmap='gray')
plt.show()
