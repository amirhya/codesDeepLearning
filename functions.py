import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


##visualization
def plot_history(history, type="everything", limits=None):
    if type == "everything":
        pd.DataFrame(history.history).plot(figsize=(10, 6))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()
    elif type == "learning rate":
        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.axis(limits)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.show()
    else:
        pass


# data processing
def normalizeData(data, scale):
    return data / scale


def splitData(data, splitRatio):
    splitIndex = int(len(data) * splitRatio)
    return data[:splitIndex], data[splitIndex:]


# Callbacks
class customizedCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\n validation/training loss: {:.2f}".format(logs["val_loss"] / logs["loss"]))


# this callback makes sure to save your model at each epoch if its the best performing on the validaiton set so far
# checkpointCallBack=keras.callbacks.ModelCheckpoint("models/imageClassificationModel.h5", save_best_only=True)


# hyperparameter tuning
def adjust_learning_rate(X, y, model, numEpochs, minLearningRate=1e-5, maxLearningRate=10,
                         optimizer=tf.keras.optimizers.Adam(), loss='mse', metric='mae'):
    def scheduler(epoch, lr):
        if not epoch:
            ##if epoch==0
            return minLearningRate
        else:
            return lr * tf.math.exp(tf.math.log(maxLearningRate / minLearningRate) / numEpochs)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[metric])
    history = model.fit(X, y, epochs=numEpochs, callbacks=[lr_schedule])
    return history
