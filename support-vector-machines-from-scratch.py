"""Support Vector Machines"""
# Aims to find the best hyperplane that separates the classes in a high dimensional space 
# Uses kernel method to handle non-linear data by mapping features into higher dimensions 
# Support Vectors - Data points outside the ϵ-margin contribute to the error and are termed "support vectors.
# Uses epsilon Tube when using SVM for Regression 

# w: weight vector
# b: bias  
# ξ, ξ*: slack variables
#   - Excess above ξ, and below ξ 
# C: Regularization parameter 
#   - Controls the trade-off between maximizing the margin and minimizing the slack variables
#   - higher value for penalizing errors more, leading to tighter fit 
#   - lower value for allowing more tolerance, leading to simpler model
# ϵ: Epsilon margin
#   - Defines margin of tolerance where deviations are not penalized 
import numpy as np 

class SupportVectorRegressor:
    def __init__(self, epsilon=0.1, C=1.0, learning_rate=0.01, num_iterations=1000):
        self.epsilon = epsilon
        self.C = C
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    

    def _epsilon_insensitive_loss(self, y, y_pred):
        """
            Compute the epsilon insensitive loss
            L = max(0, error_distance - epsilon)
            where 
                - epsilon is the margin of tolerance 
                - error_distance is distance bw actual and predicted value
            e.g, 
                - if error_distance > epsilon, loss = error_distance - epsilon
                - if error_distance < epsilon, loss = 0

            Only points outside the margin contribute to loss
        """
        return np.maximum(0, np.abs(y - y_pred) - self.epsilon)

    
    def _initialize_weights(self, n_features): 
        """
            Initialize weights and bias 
        """
        self.w = np.zeros(n_features)
        self.b = 0


    def _compute_gradients(self, X, y):
        """ 
            Compute Gradients of the loss with respect to w and b 
            - Prediction: y_pred = x * w + b 
            - Loss formula: L = max(0, error_distance - epsilon)
            - Gradients: 
                dw = - (1/m) * X.T @ (mask * sign(y_pred - y)) + w / C
                db = - (1/m) * sum(mask * sign(y_pred - y))
        """
        m = X.shape[0]
        y_pred = np.dot(X, self.w) + self.b 
        loss = self._epsilon_insensitive_loss(y, y_pred)
        
        # Mask for support Vector: Only points with Non-zero loss contribute to the gradient
        mask = (loss > 0).astype(int)

        # Gradients
        dw = -np.dot(X.T, mask * np.sign(y_pred - y)) / m + self.w / self.C
        db = -np.sum(mask * np.sign(y_pred - y)) / m 

        return loss.mean(), dw, db 

    
    def _fit(self, X, y):
        """
            Train the SVR model 
            - Initialize weights and bias 
            - iterate by updating weights 
                w = w - learning_rate * dw 
                b = b - learning_rate * db
        """
        n_features = X.shape[1]
        self._initialize_weights(n_features)

        for _ in range(self.num_iterations):
            loss, dw, db = self._compute_gradients(X, y)
            
            # update weights and bias 
            self.w -= self.learning_rate * dw 
            self.b -= self.learning_rate * db

            # log loss every 100 iterations 
            if _ % 100 == 0:
                print(f'Iteration: {_}, Loss: {loss:.4f}')
            
        
    def _predict(self, X): 
        """
            Predict the target values 
        """
        return np.dot(X, self.w) + self.b 



# Main function 
if __name__ == '__main__':
    # Sample data
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([1.5, 2.5, 3.7, 3.9])
    X_test = np.array([[1.5], [3.5]])

    # init model 
    svr = SupportVectorRegressor()

    # fit the model
    svr._fit(X_train, y_train)

    # predict 
    predictions = svr._predict(X_test)
    print(predictions)