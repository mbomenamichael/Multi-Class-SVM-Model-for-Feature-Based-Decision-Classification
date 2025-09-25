# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
# References

import numpy as np
 
class Classifier:
    def __init__(self):
        self.W = None  # Weight matrix for SVM
        self.bias = None

    def reset(self):
        # Resets the classifier
        self.W = None
        self.bias = None
    
    def fit(self, data, target, learning_rate=0.001, epochs=1000, margin=1.0, reg_strength=0.01):
        
        # This trains an SVM classifier using a hinge loss function with stochastic gradient descent, 
        # for the parameters the data is used as the the training feature vectors,
        # the target is the training labels 0-3 according to directions, 
        # the learning rate is small to avoid over corrections, 1000 iterations, margin is 1 at standard
        # reglarisation strength to prevent overfitting is is 0.01
        
        # Converts data and target into numpy arrays for fast calculations and gets number of samples and features.
        data = np.array(data)
        target = np.array(target)
        num_samples, num_features = data.shape
        num_classes = 4  # 4 possible moves
        
        # Converts labels to one-hot encoding
        y = np.eye(num_classes)[target]
        
        # Randomly initialise weights with small values and bias starts at 0
        self.W = np.random.randn(num_classes, num_features) * 0.01
        self.bias = np.zeros((num_classes, 1))
        
        for epoch in range(epochs): 
            for i in range(num_samples): # Loops through all training samples, each epoch will improve the accuracy
                x_i = data[i].reshape(-1, 1) # Extracts feature vector for each instance 
                y_i = y[i].reshape(-1, 1) # Extracts feature 
                
                scores = np.dot(self.W, x_i) + self.bias # Calculates weight x instance + bias for all 4 possible moves
                correct_class_score = np.dot(self.W[target[i]], x_i) + self.bias[target[i]] # Get the correct class score
                
                loss_gradient = np.zeros_like(self.W) # Initialise gradient storage for weight updates
                                                      # Hinge loss ensures correct moves are far enough ahead of incorrect moves.
                
                for j in range(num_classes):  # Loops through all possible moves 
                    if j == target[i]:  # Skips correct moves only care about possible moves
                        continue
                    margin_violation = scores[j] - correct_class_score + margin # Calculates margin
                    if margin_violation > 0: # If margin is violated, incorrect class is too close to the correct move, apply penalty, update gradients 
                        loss_gradient[j] += x_i.flatten() # Increase the weight of the incorrect move
                        loss_gradient[target[i]] -= x_i.flatten() # Reward correct move 
                
                # Update weights using gradient descent
                self.W -= learning_rate * (loss_gradient + reg_strength * self.W)
                self.bias -= learning_rate * loss_gradient.mean(axis=1, keepdims=True)
            
    def predict(self, data, legal=None):
        # Paramters: feature vectos, legal moves
    
        data = np.array(data).reshape(-1, 1) # Converts input feature vectors to numpy array and reshape for calculations
        scores = np.dot(self.W, data) + self.bias  # Compute scores for all possible moves
        predicted_move = np.argmax(scores) # Selects the move with the highest score 
        
        # Ensure the move is legal
        if legal and predicted_move not in legal:
            return np.random.choice(legal)
        return predicted_move
