"""Polynomial Regression"""
# Extension of linear regression to handle non-linear relationships
# Goal is to find a best fit curve 
# Prone to overfitting in case of high degree polynomials 


class PolynomialRegression: 
    # Polynomial regression 
    # y = m0 + m1(x1) + m2(x2)**2 + m3(x3)**3 + ... + mn(xn)**n
    def __init__(self, degree: int):
        self.degree = degree
        self.weights = None 

    def _transform(self, X):
        #  transform features into polynomial features 
        return [[x**d for d in range(self.degree + 1)] for x in X]

    def update_weights(self, X, y):
        # Using OLS method to update the weights 
        # m = (X^T * X)^-1 * X^T * y
        # X = [1, x, x^2, x^3, ... x^n]

        # Steps 
        # 1. Transform features into polynomial features 
        X_poly = self._transform(X)
        # 2. Compute transpose 
        X_T = [[row[i] for row in X_poly] for i in range(len(X_poly[0]))]
        # 3. X_T * x 
        X_T_X = [[sum(a * b for a, b in zip(X_T_row, X_poly_col)) for X_poly_col in zip(*X_poly)] for X_T_row in X_T]
        # 4. X_T * y
        X_T_y = [sum(X_T_row[i] * y[i] for i in range(len(y))) for X_T_row in X_T]
        # 5. Inverse of X_T_X 
        # Solve for coefficients (X^T * X)^-1 * (X^T * y) (for simplicity, using a 2x2 inversion)
        if len(X_T_X) == 2:  # Only works for 2x2 matrices
            det = X_T_X[0][0] * X_T_X[1][1] - X_T_X[0][1] * X_T_X[1][0]
            inv_X_T_X = [
                [X_T_X[1][1] / det, -X_T_X[0][1] / det],
                [-X_T_X[1][0] / det, X_T_X[0][0] / det]
            ]
            self.coefficients = [sum(inv_X_T_X[i][j] * X_T_y[j] for j in range(2)) for i in range(2)]
        else:
            raise NotImplementedError("Matrix inversion for larger degrees is not implemented")

    def predict(self, X):
        """
        Predict using the learned polynomial regression model.
        """
        # Transform X into polynomial features
        X_poly = self._transform(X)
        # Compute predictions
        return [sum(c * x for c, x in zip(self.coefficients, row)) for row in X_poly]


# Example
X = [1, 2]  # Features
y = [1.2, 4.1] # Target 

# Create a polynomial regression model
degree = 2
model = PolynomialRegression(degree=degree)
model.update_weights(X, y)
print(
    f"Model Prediction for X: {model.predict(X)}"
)
