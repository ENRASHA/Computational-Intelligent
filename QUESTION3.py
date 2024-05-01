•	from keras.models import Sequential
•	from keras.layers import Dense
•	import numpy as np
•	
•	# Define the truth table data (p, q, r) as input and A as output
•	X = np.array([
•	    [1, 1, 1],  # T T T
•	    [1, 1, 0],  # T T F
•	    [1, 0, 1],  # T F T
•	    [1, 0, 0],  # T F F
•	    [0, 1, 1],  # F T T
•	    [0, 1, 0],  # F T F
•	    [0, 0, 1],  # F F T
•	    [0, 0, 0],  # F F F
•	])
•	
•	# Output corresponding to the given truth table for A
•	y = np.array([1, 0, 1, 0, 1, 0, 0, 0])  # T=1 and F=0
•	
•	# Define the 3-layer neural network
•	model = Sequential()
•	model.add(Dense(4, input_dim=3, activation='relu'))  # Hidden layer with 4 neurons
•	model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron
•	
•	# Compile the model
•	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
•	
•	# Train the model on the data
•	model.fit(X, y, epochs=1000, verbose=0)
•	
•	# Evaluate the model on the truth table
•	accuracy = model.evaluate(X, y, verbose=0)
•	predictions = model.predict(X)
•	
•	# Output the accuracy and predictions
•	accuracy, predictions.round()
•	
