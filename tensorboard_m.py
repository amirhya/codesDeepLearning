import os
import time

import keras.callbacks


class TensorBoardCB:
    def __init__(self, rootLogDir=os.path.join(os.curdir, "myLogs"), modelName=""):
        self.rootDirectory = rootLogDir
        run_id = time.strftime("Model_"+modelName+"_run_%Y_%m_%d:%H_%M_%S")
        self.runLogDir = os.path.join(self.rootDirectory, run_id)

    def getCB(self, writeGrads=False):
        return keras.callbacks.TensorBoard(self.runLogDir,write_grads = writeGrads)


# TODO: Write code to visulize and track gradient weights