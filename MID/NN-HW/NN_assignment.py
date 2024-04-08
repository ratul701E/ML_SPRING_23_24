import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder


class MultiClassNeuralNetwork(object):
    def __init__(self):
        # Define the architecture of the neural network
        input_neurons = 2
        hidden_neurons_1 = 20
        hidden_neurons_2 = 15
        hidden_neurons_3 = 10
        output_neurons = 5  # Number of classes
        
        # Set the learning rate
        self.learning_rate = 0.03
        
        # Initialize weights with random values
        self.W_input_hidden_1 = np.random.randn(input_neurons, hidden_neurons_1)
        self.W_hidden_1_hidden_2 = np.random.randn(hidden_neurons_1, hidden_neurons_2)
        self.W_hidden_2_hidden_3 = np.random.randn(hidden_neurons_2, hidden_neurons_3)
        self.W_hidden_3_output = np.random.randn(hidden_neurons_3, output_neurons)
        
    def softmax(self, x):
        # Softmax activation function to obtain class probabilities
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, Y, pred):
        # Cross-entropy loss function to measure the difference between predicted and true class probabilities
        m = Y.shape[0]
        loss = -1/m * np.sum(Y * np.log(pred))
        return loss
    
    def sigmoid(self, x, der=False):
        # Sigmoid activation function for hidden layers
        if der == True:
            return x * (1-x)
        else:
            return 1 / (1 + np.exp(-x))
        
    def feedForward(self, X):
        # Feedforward propagation through the network
        # Calculate activations of hidden layers
        hidden_input_1 = np.dot(X, self.W_input_hidden_1)
        self.hidden_output_1 = self.sigmoid(hidden_input_1)
        
        hidden_input_2 = np.dot(self.hidden_output_1, self.W_hidden_1_hidden_2)
        self.hidden_output_2 = self.sigmoid(hidden_input_2)
        
        hidden_input_3 = np.dot(self.hidden_output_2, self.W_hidden_2_hidden_3)
        self.hidden_output_3 = self.sigmoid(hidden_input_3)
        
        # Calculate activations of output layer
        output_input = np.dot(self.hidden_output_3, self.W_hidden_3_output)
        pred = self.softmax(output_input)
        return pred
    
    def backPropagation(self, X, Y, pred):
        # Back propagation to update weights based on prediction error
        output_error = Y - pred
        output_delta = self.learning_rate * output_error
        
        hidden_error_3 = output_delta.dot(self.W_hidden_3_output.T) * self.sigmoid(self.hidden_output_3, der=True)
        hidden_delta_3 = self.learning_rate * hidden_error_3
        
        hidden_error_2 = hidden_delta_3.dot(self.W_hidden_2_hidden_3.T) * self.sigmoid(self.hidden_output_2, der=True)
        hidden_delta_2 = self.learning_rate * hidden_error_2
        
        hidden_error_1 = hidden_delta_2.dot(self.W_hidden_1_hidden_2.T) * self.sigmoid(self.hidden_output_1, der=True)
        hidden_delta_1 = self.learning_rate * hidden_error_1
        
        # Update weights
        self.W_input_hidden_1 += X.T.dot(hidden_delta_1)
        self.W_hidden_1_hidden_2 += self.hidden_output_1.T.dot(hidden_delta_2)
        self.W_hidden_2_hidden_3 += self.hidden_output_2.T.dot(hidden_delta_3)
        self.W_hidden_3_output += self.hidden_output_3.T.dot(output_delta)
        
    def train(self, X, Y, epochs=30000):
        # Training the neural network
        err = []
        for i in range(epochs):
            output = self.feedForward(X)
            # Calculate and store the training error for each epoch
            err.append(self.cross_entropy_loss(Y, output))
            self.backPropagation(X, Y, output)
        return err

# Generating synthetic dataset
np.random.seed(420)

num_samples_per_class = 100
centers = [(2, 2), (2, -2), (-2, 2), (-2, -2), (0, 0)]

X = np.concatenate([np.random.randn(num_samples_per_class, 2) + center for center in centers])
Y = np.eye(5)[np.repeat(np.arange(5), num_samples_per_class)]

# Shuffling the dataset
permutation = np.random.permutation(X.shape[0])
X, Y = X[permutation, :], Y[permutation, :]


# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initializing the neural network
multi_class_NN = MultiClassNeuralNetwork()

# Training the neural network
training_error = multi_class_NN.train(X_train, Y_train)

# Plotting the training error
plt.plot(training_error)
plt.title('Training Error')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.savefig("TE.jpg")
plt.show()

# Testing the neural network with testing data
predictions = multi_class_NN.feedForward(X_test)

# Converting predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(Y_test, axis=1)

# Calculating evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average=None)
recall = recall_score(true_labels, predicted_labels, average=None)
f1 = f1_score(true_labels, predicted_labels, average=None)

# Printing evaluation metrics
print(f"\nEvaluation Metrics:\n\nAccuracy: {accuracy}\n\nPrecision: {precision}\n\nRecall: {recall}\n\nF1-score: {f1}\n\n")

# Plotting confusion matrix
cmf = confusion_matrix(true_labels, predicted_labels)
xylables = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]
plt.figure()
sns.heatmap(cmf, annot=True, fmt="d", cmap="Greens", xticklabels=xylables, yticklabels=xylables)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("CFM.jpg")
plt.show()
