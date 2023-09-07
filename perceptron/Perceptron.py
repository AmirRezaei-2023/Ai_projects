import numpy as np
from sklearn.metrics import accuracy_score
class Perceptron:
    def __init__(self):
        self.weights = None

    def weighting(self, input):
        return np.dot(input, self.weights)

    def activation(self, weighted_input):
        
        # sign activation function
        if weighted_input >= 0:
            return 1
        else:
            return -1

    def predict(self, inputs):
        
        # adding a 1 to the first position of each input (adding the bias term)
        new_inputs = np.insert(inputs, 0, [1], axis=1)

        # a list of final prediction for each test sample
        predictions = []

        for input in new_inputs:
            
            weighted_input = self.weighting(input)
            prediction = self.activation(weighted_input)
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
            for sample, target in zip(new_inputs, outputs):
                weighted_input = self.weighting(sample)
                diff =  target - self.activation(weighted_input)
                self.weights = self.weights + learning_rate * diff * sample
    
            print('Epoch #' + str(epoch) + ' - Accuracy: ' + str(accuracy_score(self.predict(inputs), outputs)))

