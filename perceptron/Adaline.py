import numpy as np
from sklearn.metrics import accuracy_score
class Adaline:
    def __init__(self):
        self.weights = None

    def weighting(self, input):
        return np.dot(input, self.weights)

    def activation(self, weighted_input):
        return weighted_input

    def predict(self, inputs):

        # adding a 1 to the first position of each input (adding the bias term)
        new_inputs = np.insert(inputs, 0, [1], axis=1)
        
        # a list of final prediction for each test sample
        predictions = []

        for input in new_inputs:
            weighted_input = self.weighting(input)
            weighted_input = self.activation(weighted_input)
            prediction = None
            if weighted_input >= 0:
                prediction = 1
            else:
                prediction = -1
            predictions.append(prediction)

        # converting the list to a numpy array
        predictions = np.array(predictions)

        return predictions

    def fit(self, inputs, outputs, learning_rate=0.1, epochs=64):

        # adding a 1 to the first position of each input (adding the bias term)
        new_inputs = np.insert(inputs, 0, [1], axis=1)

        # initializing the weights
        self.weights = np.random.rand(new_inputs.shape[1])

        # training loop
        for epoch in range(epochs):
            weighted_input = self.weighting(new_inputs)
            diff =  outputs - self.activation(weighted_input)
            self.weights = self.weights + learning_rate * new_inputs.T.dot(diff)
    
            print('Epoch #' + str(epoch) + ' - Accuracy: ' + str(accuracy_score(self.predict(inputs), outputs)))

