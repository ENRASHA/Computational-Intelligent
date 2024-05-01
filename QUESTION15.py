# Training data for a 3-class classification problem with 3 input features
X_train = np.array([[1, 2, 1], [2, 1, 3], [1, -1, -2], [2, -1, 0], [-1, 1, 2], [-2, 2, 0]])
y_train = np.array([1, 1, 2, 2, 3, 3])  # Class labels: 1 for Class 1, 2 for Class 2, 3 for Class 3

# Convert class labels to one-hot encoding since NN outputs are generally one-hot encoded for classification
lb = LabelBinarizer()
Y_train = lb.fit_transform(y_train)

# Define a neural network model with 3 inputs and 3 outputs
nn_clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=10000, random_state=42)

# Train the neural network on the data
nn_clf.fit(X_train, Y_train)

# Test vector [2, 0, 1]
X_test = np.array([[2, 0, 1]])

# Predict the class for the test vector
y_pred_test = nn_clf.predict(X_test)

# Invert the one-hot encoding to get back the class label
predicted_class = lb.inverse_transform(y_pred_test)

# Output the predicted class for the test vector
predicted_class
