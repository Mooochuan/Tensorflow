# tensorflow2.0  初学者
import tensorflow as tf
import numpy as np
# load minst data
mnist = tf.keras.datasets.mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
print(xtrain.shape, ytrain.shape, xtest.shape,ytest.shape)
# The data is normalized and the values are scaled to [0,1]
xtrain, xtest = xtrain/255.0, xtest/255.0

# build the model by stacking layer
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
#
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# tf.losses.loss
# save the model
check_path = 'my_mnist_checkpoint/ckpt/cp-{epoch:04d}.ckpt'
save_mode_cp = tf.keras.callbacks.ModelCheckpoint(check_path, verbose=1, save_weights_only=True, period=2)

# train
model.fit(xtest,ytest,epochs=5,callbacks=[save_mode_cp])
