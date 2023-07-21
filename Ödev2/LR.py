class LinearRegression:
# sınıf değişkenlerini başlat
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.m1 = 1
        self.m2 = 2
        self.bias = 0
        self.loss = []
        self.accuracy = []    
    # hipotez fonksiyonu p(x) = m1 * x + m2 * x + bias
    def hypothesis(self, weight, height):
        return self.m1 * height + self.m2 * weight + self.bias
    # Verilen veri kümesi (boy, kilo, bmi) ile modeli eğitmek için fit fonksiyonu
    def fit(self, height, weight, bmi):
        for epoch in range(self.num_epochs):
            bias_gradient = 0
            predictions = [self.hypothesis(x, y) for x, y in zip(weight, height)]
            # Gradyanları hesapla
            weight_gradient = sum(2 * (predictions[i] - bmi[i]) * weight[i] for i in range(len(weight))) / len(weight)
            height_gradient = sum(2 * (predictions[i] - bmi[i]) * height[i] for i in range(len(height))) / len(height)
            bias_gradient = sum(2 * (predictions[i] - bmi[i]) for i in range(len(weight))) / len(weight)
            self.m2 -= self.learning_rate * weight_gradient
            self.m1 -= self.learning_rate * height_gradient
            self.bias -= self.learning_rate * bias_gradient
    # Verilen boy ve kilo değerleriyle bmi değerlerini tahmin etmek için predict fonksiyonu
    def predict(self, new_height, new_weight):
        predictions = []
        for i in range(len(new_height)):
            prediction = self.hypothesis(new_weight[i], new_height[i])
            if self.num_epochs % 500 == 0: 
                print("Epoch: {} - Prediction: {}".format(self.num_epochs, prediction))
            predictions.append(prediction)
        return predictions
