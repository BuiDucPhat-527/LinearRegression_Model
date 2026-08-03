from main import LinearRegressionScratch
import numpy as np

#train
X = np.random.rand(100, 1)
y = 3 * X.flatten() + 4 + np.random.randn(100) * 0.1

model = LinearRegressionScratch(lr=0.1, iterations=1000)
model.fit(X, y)

#save
model.save_model("trained_linear_model.pkl")