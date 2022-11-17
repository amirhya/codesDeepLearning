"""
FashionMNIST

Functional API dev
Sequential API dev





"""


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import functions

print("Tensorflow version:" + tf.__version__)
print("Keras Version:"+ keras.__version__)

fashionMnist = keras.datasets.fashion_mnist
classNames = ['T-shirt/top', 'Trouser', 'Pullover', "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
(XTrain, yTrain), (XTest, yTest) = fashionMnist.load_data()
splitRatio = 0.8
scale = 255.0

# preprocessing
XTrain = functions.normalizeData(data=XTrain, scale=scale)
XTest = functions.normalizeData(data=XTest, scale=scale)
XTrain, XValid = functions.splitData(data=XTrain, splitRatio=splitRatio)
yTrain, yValid = functions.splitData(data=yTrain, splitRatio=splitRatio)


#
# print(fashionMnist.shape)
# print(fashionMnist.dtype)

class ModelSequential:
    def __init__(self):
        self.model = keras.models.Sequential()

    def build(self):
        self.model.add(keras.layers.Flatten(input_shape=(28, 28)))
        self.model.add(keras.layers.Dense(300, activation='relu'))
        self.model.add(keras.layers.Dense(100, activation=keras.activations.relu))
        self.model.add(keras.layers.Dense(10, activation='softmax'))

    def compile(self):
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])

    def fit(self, XTrain, yTrain, XValid, yValid, **kwargs):
        history = self.model.fit(XTrain, yTrain, validation_data=(XValid, yValid), **kwargs)
        return history


class ModelFunctional:
    def __init__(self):
        pass
    def build(self):
        input = keras.layers.Input(shape=(28, 28), name='input')
        flatten = keras.layers.Flatten()(input)
        hidden1 = keras.layers.Dense(300, activation='relu')(flatten)
        hidden2 = keras.layers.Dense(100, activation='relu')(hidden1)
        output = keras.layers.Dense(10, activation='softmax')(hidden2)
        self.model = keras.Model(inputs=[input], outputs=[output])

    def compile(self):
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])

    def fit(self, XTrain, yTrain, XValid, yValid, callbacks=[], **kwargs):
        history = self.model.fit(XTrain, yTrain, validation_data=(XValid, yValid), callbacks=callbacks, **kwargs)
        return history




##this callback makes sure to save your model at each epoch if its the best performing on the validaiton set so far
checkpointCallBack = keras.callbacks.ModelCheckpoint("models/imageClassificationModel.h5", save_best_only=True)

modelType = "sequential"

if modelType == "sequential":
    model = ModelSequential()
elif modelType == "functional":
    mode = ModelFunctional()

# model=myModelFunctional()
model.build()
model.compile()
customizedCallBack = customizedCallBack()
history = model.fit(XTrain, yTrain, XValid=XValid, yValid=yValid, callbacks=[checkpointCallBack, customizedCallBack],
                    epochs=30)
