from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import functions
import tensorflow as tf
from tensorflow import keras

data = fetch_california_housing()

X, XTest, y, yTest = train_test_split(data.data, data.target)
XTrain, XValid, yTrain,yValid= train_test_split(X, y)

scaler=StandardScaler()
scaler.fit(XTrain)
XTrain=scaler.transform(XTrain)
XTest=scaler.transform(XTest)
XValid=scaler.transform(XValid)



class myModelSequential():
    def __init__(self, input_shape):
        self.model = keras.models.Sequential()
        self.input_shape=input_shape
    def build(self):
        self.model.add(keras.layers.Input(shape=self.input_shape))
        self.model.add(keras.layers.Dense(300,activation='relu', kernel_initializer= "he_normal"))
        self.model.add(keras.layers.Dense(100, activation=keras.activations.relu,kernel_initializer= "he_normal"))
        self.model.add(keras.layers.Dense(1, activation='linear'))
    def compile(self, learningRate=0.001):
        self.model.compile(loss="mse",
                           optimizer=keras.optimizers.Adam(lr=learningRate))
    def fit(self, XTrain, yTrain, XValid, yValid, callbacks=[], **kwargs):
        history=self.model.fit(XTrain,yTrain,validation_data=(XValid,yValid),callbacks=callbacks,**kwargs)
        return history
    def evaluate(self,X, y):
        return self.model.evaluate(X,y)
    def predict(self,X):
        return self.model.predict(X)
    def getModel(self):
        return self.model


model=myModelSequential(XTrain.shape[1:])
model.build()

rawModel=model.getModel()
lrHistory=functions.adjustLearningRate(X=XTrain,y=yTrain, model=rawModel, numEpochs=500, minLearningRate=1e-5, maxLearningRate=10,
                       optimizer=keras.optimizers.Adam(), loss='mse',metric='mae')

model.compile()


history=model.fit(XTrain,yTrain, XValid, yValid, epochs=100)
yPred= model.predict(XTest)
plt.plot(yPred-yTest,label='pred')
plt.show()


