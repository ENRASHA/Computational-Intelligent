6.	from tensorflow.keras.models import Sequential
7.	from tensorflow.keras.layers import SimpleRNN, Dense
8.	from tensorflow.keras.optimizers import Adam
9.	
10.	# For demonstration, generate synthetic sequential data
11.	# Assuming each sequence has 3 time steps and there are 100 samples
12.	time_steps = 3
13.	features = 2
14.	samples = 100
15.	
16.	# Generate random sequential data
17.	X = np.random.rand(samples, time_steps, features)
18.	# Generate random labels
19.	y = np.random.randint(0, 2, size=(samples, 1))
20.	
21.	# Split the data into 80% train and 20% test
22.	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
23.	
24.	# Create an RNN model
25.	model = Sequential()
26.	model.add(SimpleRNN(5, input_shape=(time_steps, features), activation='tanh'))
27.	model.add(Dense(1, activation='sigmoid'))
28.	
29.	# Compile the model
30.	model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
31.	
32.	# Train the RNN on the training data
33.	history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)
34.	
35.	# Evaluate the RNN on the test data
36.	test_loss, test_accuracy = model.evaluate(X_test, y_test)
37.	
38.	# Output the performance metric
39.	test_accuracy
