# Defining the dataset based on the provided input vectors and their corresponding classes
X = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0], # S1 - Class 1
    [0, 1, 1, 0, 0, 0, 0, 0], # S2 - Class 1
    [1, 0, 1, 0, 0, 0, 0, 0], # S3 - Class 1
    [1, 1, 0, 0, 0, 0, 0, 0], # S4 - Class 1
    [0, 0, 0, 1, 0, 0, 0, 0], # S5 - Class 2
    [0, 0, 0, 0, 1, 1, 0, 0], # S6 - Class 2
    [0, 0, 0, 0, 1, 0, 1, 0], # S7 - Class 2
    [0, 0, 0, 1, 0, 1, 0, 0], # S8 - Class 2
    [0, 0, 0, 0, 0, 0, 1, 1], # S9 - Class 3
    [0, 0, 0, 0, 0, 1, 1, 1], # S10 - Class 3
    [0, 0, 0, 0, 1, 1, 0, 0], # S11 - Class 3
    [0, 0, 0, 0, 0, 1, 1, 0]   # S12 - Class 3
])

# Labels for the dataset
y = np.array([
    'Class 1', 'Class 1', 'Class 1', 'Class 1',  # Class 1 labels
    'Class 2', 'Class 2', 'Class 2', 'Class 2', # Class 2 labels
    'Class 3', 'Class 3', 'Class 3', 'Class 3'   # Class 3 labels
])

# Encode the class labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Initialize the neural network classifier
# Since we have 3 classes, we'll add an output layer with 3 units
clf = MLPClassifier(hidden_layer_sizes=(10,), random_state=1, max_iter=300)

# Train the classifier
clf.fit(X, y_encoded)

# Make predictions (using the same data in this example)
y_pred = clf.predict(X)

# Generate a classification report
report = classification_report(y_encoded, y_pred, target_names=le.classes_)
conf_matrix = confusion_matrix(y_encoded, y_pred)

(report, conf_matrix)


Confusion Matrix :
[[4, 0, 0], # Class 1: All 4 samples correctly classified
 [0, 4, 0], # Class 2: All 4 samples correctly classified
 [0, 1, 3]] # Class 3: 3 samples correctly classified, 1 misclassified as Class 2


