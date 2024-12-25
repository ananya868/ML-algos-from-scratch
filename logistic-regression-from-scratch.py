"""Logistic Regression"""
# Finds the best line that separates the classes in a 2D space 
# Pull the line towards the wrongly classified points, and push away from correctly classified points 
# Uses sigmoid function to compute probabilities (0 - 1)
# Uses log-loss function to compute the error
import math


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Compute probabilities
def predict_proba(weights, bias, X):
    # y = m0 + m1*x1 + m2*x2 + ... + mn*xn
    # y_pred = sigmoid(y)
    return [sigmoid(sum(weights[j] * X[i][j] for j in range(len(weights))) + bias) for i in range(len(X))]

# j -> iterate over the features (x1, x2 ... xn)
# i -> iterate over the samples (rows) 
# Compute gradients
def compute_gradients(weights, bias, X, y):
    n_samples = len(X) # Rows
    n_features = len(weights) # Features or len(X[0])
    y_pred = predict_proba(weights, bias, X)
    errors = [y_pred[i] - y[i] for i in range(n_samples)]
    weight_gradients = [sum(errors[i] * X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
    bias_gradient = sum(errors) / n_samples
    return weight_gradients, bias_gradient

# Train model
def train_logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    for _ in range(num_iterations):
        weight_gradients, bias_gradient = compute_gradients(weights, bias, X, y)
        weights = [weights[j] - learning_rate * weight_gradients[j] for j in range(n_features)]
        bias -= learning_rate * bias_gradient
    return weights, bias

# Predict class labels
def predict(weights, bias, X, threshold=0.5):
    probabilities = predict_proba(weights, bias, X)
    return [1 if p >= threshold else 0 for p in probabilities]

# Example Usage
X = [[0.5], [1.5], [2.0], [3.0], [4.0]]  # Features
y = [0, 0, 1, 1, 1]  # Labels

# Train model
weights, bias = train_logistic_regression(X, y, learning_rate=0.1, num_iterations=1000)

# Make predictions
probabilities = predict_proba(weights, bias, X)
predictions = predict(weights, bias, X)

print("Learned Weights:", weights)
print("Learned Bias:", bias)
print("Probabilities:", probabilities)
print("Predicted Classes:", predictions)

