"""Linear Regression""" 
# Supervised learning algorithm that models the relationship between a scalar dependent variable y and one or more independent variables denoted X.
# The goal is to find the best-fitting linear line that minimizes the difference bw actual and predicted values.
# Assumes a linear relation and sensitive to outliers 


class LinearRegression: 
    # Simple Linear Regression model
    # y = m0 + m1(x1) + m2(x2) ... + mn(xn)
    def __init__(self, x: list[float], y: list[float], m: float = 0, c: float = 0):
        self.m = m
        self.c = c
        self.x = x
        self.y = y

    def update_weights(self):
        # Using Ordinary Least Squares (OLS) method to find the best fit line
        # m = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)^2
        # c = ȳ - m * x̄
        # compute mean of x and y
        mean_x = sum(self.x) / len(self.x)
        mean_y = sum(self.y) / len(self.y)

        # compute 'm' and 'c'
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(self.x, self.y))
        denominator = sum((xi - mean_x) ** 2 for xi in self.x) 
        self.m = numerator / denominator 
        self.c = mean_y - self.m * mean_x 

        return self.m, self.c 
    
    def predict(self, feature: int):
        return self.m * feature + self.c
    

class LinearRegressionGD:
    # Gradient Descent (GD) method to find the best fit line
    # y = m0 + m1(x1) + m2(x2) ... + mn(xn)
    def __init__(self, x: list[float], y: list[float], m: float = 0, c: float = 0, alpha: float = 0.01, epochs: int = 100):
        self.m = m
        self.c = c
        self.x = x
        self.y = y
        self.alpha = alpha
        self.epochs = epochs

    def update_weights(self):
        # Update the weights using: 
        # m = m - α * ∂(J(m, c)) / ∂m 
        # c = c - α * ∂(J(m, c)) / ∂c 
        for _ in range(self.epochs):
            self.m = self.m - self.alpha * ((-2 / len(self.x)) * sum(xi * (yi - (self.m * xi + self.c)) for xi, yi in zip(self.x, self.y)))  
            self.c = self.c - self.alpha * ((-2 / len(self.x)) * sum(yi - (self.m * xi + self.c) for xi, yi in zip(self.x, self.y)))
        
        return self.m, self.c  
    
    def predict(self, feature: int): 
        return round(self.m * feature + self.c, 3)



if __name__ == "__main__":
    # Example usage
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    # OLS Method
    model = LinearRegression(x, y) 
    model.update_weights() 
    print("::::Using OLS Method::::")
    print(
        f"Slope: {model.m} \nIntercept: {model.c}"
    )
    print(
        f"Prediction for x = 6: {model.predict(6)}\n"
    )

    # Gradient Descent Method 
    model = LinearRegressionGD(x, y)
    model.update_weights()
    print("::::Using Gradient Descent Method::::")
    print(
        f"Slope: {model.m} \nIntercept: {model.c}"
    )
    print(
        f"Prediction for x = 6: {model.predict(6)}"
    )







