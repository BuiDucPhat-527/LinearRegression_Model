import numpy as np
import matplotlib.pyplot as plt
import joblib

class LinearRegressionScratch:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr=lr
        self.iterations=iterations
        self.weights=None
        self.bias=None
        self.loss_history=[]

    def fit(self, X, y):
        n_samples, n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.iterations):
            y_predicted=np.dot(X, self.weights) +self.bias

            loss=np.mean((y_predicted-y)**2)
            self.loss_history.append(loss)

            dw=(1/n_samples)*np.dot(X.T, (y_predicted-y))
            db=(1/n_samples)*np.sum(y_predicted-y)

            self.weights-=self.lr*dw
            self.bias-=self.lr*db

    def predict(self,X):
        return np.dot(X, self.weights)+self.bias
    
    def save_model(self, filepath):
        # Lưu cặp w và b vào một file
        model_data = {'weights': self.weights, 'bias': self.bias}
        joblib.dump(model_data, filepath)
        print(f"Saved model to {filepath}")

    def load_model(self, filepath):
        # Tải tham số từ file và nạp vào model
        model_data = joblib.load(filepath)
        self.weights = model_data['weights']
        self.bias = model_data['bias']
        print(f"Loaded model from {filepath}")