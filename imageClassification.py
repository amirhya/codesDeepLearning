import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

fashionMnist = keras.datasets.fashion_mnist
classNames = ['T-shirt/top','Trouser', 'Pullover',"Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
(XTrain, yTrain), (XTest, yTest) =fashionMnist.load_data()
splitRatio=0.8
scale=255.0

def normalizeData(data,scale):
    return data/scale

def splitData(data,splitRatio):
    splitIndex=int(len(data)*splitRatio)
    return data[:splitIndex],data[splitIndex:]

##preprocessing
XTrain = normalizeData(data=XTrain,scale=scale)
XTest = normalizeData(data=XTest,scale=scale)
XTrain, XValid =splitData(data=XTrain,splitRatio=splitRatio)
yTrain,yValid=splitData(data=yTrain,splitRatio=splitRatio)
#
# print(fashionMnist.shape)
# print(fashionMnist.dtype)

class myModelSequential():
    def __init__(self):
        self.model = keras.models.Sequential()
    def build(self):
        self.model.add(keras.layers.Flatten(input_shape=(28,28)))
        self.model.add(keras.layers.Dense(300,activation='relu'))
        self.model.add(keras.layers.Dense(100, activation=keras.activations.relu))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
    def compile(self):
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer=keras.optimizers.Adam(lr=0.001),metrics=["accuracy"])
    def fit(self, XTrain, yTrain, XValid, yValid, **kwargs):
        history=self.model.fit(XTrain,yTrain,validation_data=(XValid,yValid),**kwargs)
        return history


class myModelFunctional():

model=myModelSequential()

# model.build()
# model.compile()
# history=model.fit(XTrain, yTrain, XValid=XValid, yValid=yValid, epochs=30)