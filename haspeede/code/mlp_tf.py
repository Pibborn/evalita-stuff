import tensorflow as tf
from tensorflow import keras
import numpy as np

def preprocess_data(X, y):
    X = np.array(X.todense())
    y = np.array(list(map(int, y)))
    y = keras.utils.to_categorical(y)
    return X, y



def create_model(layer_sizes, lr=1e-5):
    model = keras.Sequential()
    for size in layer_sizes:
        model.add(keras.layers.Dense(size, activation='relu'))
    model.compile(optimizer=tf.train.AdamOptimizer(lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def train(model, X_train, y_train, X_test, y_test, num_epochs=1000, batch_size=32, run_id=None):
    if run_id == None:
        run_id = 'default'
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/'+run_id, histogram_freq=1, write_graph=True,
                              write_images=False)
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
              validation_data=(X_test, y_test),
              shuffle=True, callbacks=[tensorboard])
