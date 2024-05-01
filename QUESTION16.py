from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming the data provided in the image is copied correctly into the arrays
X = np.array([
    [-7.7947e-01, 8.3822e-01],
    [1.5563e-01, 8.9537e-01],
    # ... (other data points) ...
    [5.0744e-01, 7.5872e-01]
])

y = np.array([
    1.0,
    1.0,
    # ... (other target values) ...
    -1.0
])

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the neural network model with 2 inputs and 1 output
# Using a hidden layer with a size of 10 neurons as an example
nn_model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=10000, random_state=42)

# Train the neural network on the training data
nn_model.fit(X_train, y_train)

# Predict the output on the test data
y_pred = nn_model.predict(X_test)

# Calculate the mean squared error and the R^2 score for performance evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the performance metrics and the model
(mse, r2, nn_model)
