from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import functions
import tensorflow as tf
from tensorflow import keras
import tensorboard_m

data = fetch_california_housing()

X, XTest, y, yTest = train_test_split(data.data, data.target)
XTrain, XValid, yTrain, yValid = train_test_split(X, y)

scaler = StandardScaler()
scaler.fit(XTrain)
XTrain = scaler.transform(XTrain)
XTest = scaler.transform(XTest)
XValid = scaler.transform(XValid)


class ModelSequential:
    def __init__(self, input_shape):
        self.model = keras.models.Sequential()
        self.input_shape = input_shape

    def build(self, kernelInitializer="he_normal", numNeurons=10,numLayers=2):
        self.model.add(keras.layers.Input(shape=self.input_shape))
        for _ in range(numLayers):
            self.model.add(keras.layers.Dense(numNeurons))
            #self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Activation("relu"))
            self.model.add(keras.layers.Dropout(0.3))

        self.model.add(keras.layers.Dense(1, activation='linear'))

    def compile(self, loss, learningRate=0.001):
        self.model.compile(loss=loss,
                           optimizer=keras.optimizers.Adam(learning_rate=learningRate))

    def fit(self, XTrain, yTrain, XValid, yValid, callbacks=[], **kwargs):
        history = self.model.fit(XTrain, yTrain, validation_data=(XValid, yValid), callbacks=callbacks, **kwargs)
        return history

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model


loss=tf.keras.losses.Huber()
model = ModelSequential(XTrain.shape[1:])
model.build(numNeurons=128,numLayers=4)

rawModel = model.get_model()
# lrHistory = functions.adjust_learning_rate(X=XTrain, y=yTrain, model=rawModel, numEpochs=200, minLearningRate=1e-6,
#                                            maxLearningRate=1, optimizer=keras.optimizers.Adam(),
#                                            loss=loss,
#                                            metric='mae')
#
# functions.plot_history(lrHistory, type="learning rate", limits=[1e-6, 1, 0, 0.5])


model.compile(learningRate=1e-3, loss=loss)
tensorboardCB= tensorboard_m.TensorBoardCB(modelName='sequemtialRegression').getCB()
checkpointCallBack=keras.callbacks.ModelCheckpoint("models/imageClassificationModel.h5", save_best_only=True)

history = model.fit(XTrain, yTrain, XValid, yValid, epochs=100, callbacks=[tensorboardCB])
model.evaluate(XTest,yTest)


# #

# functions.plot_history(history)
# yPred = model.predict(XTest)
# plt.plot(yPred - yTest, label='pred')
# plt.show()
#
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.hist(yPred)
#
# # Show plot
# plt.show()
