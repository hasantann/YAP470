class LinearRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_slope = 0
        self.intercept = 0

    def normalize(self, feature):
        return (feature - min(feature)) / (max(feature) - min(feature))

    def hypothesis(self, x):
        return self.weight_slope * x + self.intercept

    def cost_function(self, predictions, targets):
        return sum((predictions - targets) ** 2) / (2 * len(targets))

    def fit(self, height, weight, bmi):
        normalized_height = self.normalize(height)
        normalized_weight = self.normalize(weight)

        for epoch in range(self.num_epochs):
            predictions = [self.hypothesis(x) for x in normalized_height]

            weight_gradient = sum((predictions[i] - bmi[i]) * normalized_height[i] for i in range(len(height))) / len(height)
            intercept_gradient = sum(predictions[i] - bmi[i] for i in range(len(height))) / len(height)

            self.weight_slope -= self.learning_rate * weight_gradient
            self.intercept -= self.learning_rate * intercept_gradient

            cost = self.cost_function(predictions, bmi)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Cost: {cost}")

    def predict(self, new_height, new_weight):
        normalized_new_height = self.normalize(new_height)
        normalized_new_weight = self.normalize(new_weight)

        predictions = [self.hypothesis(x) for x in normalized_new_height]
        return predictions