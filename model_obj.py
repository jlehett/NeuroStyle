import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

class SketchModel:
    def __init__(self, input_shape):
        self.model = None
        if input_shape != None:
            self.input_shape = (input_shape[1],input_shape[2])

    def buildModel(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        self.model = model

    def trainModel(self, x_train, y_train, batch_size=32, epochs=100):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    def saveModel(self, filename):
        self.model.save(filename)

    def loadModel(self, filename):
        try:
            self.model = load_model(filename)
            if self.model == None:
                self.model = self.buildModel()
        except:
            self.buildModel()