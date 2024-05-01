class DigitClassifierNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases for the hidden layer
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Initialize weights and biases for the output layer
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=1, keepdims=True)

    def forward(self, X):
        # Forward pass through the first layer (input layer to hidden layer)
        Z1 = X.dot(self.W1) + self.b1
        A1 = self.relu(Z1)
        
        # Forward pass through the second layer (hidden layer to output layer)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = self.softmax(Z2)
        
        return A2

    def compute_loss(self, Y_hat, Y):
        # Cross-entropy loss function
        m = Y.shape[0]
        log_likelihood = -np.log(Y_hat[range(m), Y])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backprop(self, X, Y, Y_hat, learning_rate):
        # Backpropagation to update weights and biases
        # ... This would include calculating the gradient and updating W1, b1, W2, b2
    
    def train(self, X, Y, epochs, learning_rate):
        # Training loop over the number of epochs
        # ... This would include calling forward, compute_loss, and backprop

    def predict(self, X):
        # Predict the class from the forward pass probabilities
        # ... This would include calling forward and returning argmax of output
