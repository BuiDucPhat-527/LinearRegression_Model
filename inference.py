from main import LinearRegressionScratch
import numpy as np

model = LinearRegressionScratch()

model.load_model("trained_linear_model.pkl")

while True:
    input_val = input("Enter x to predict (or 'q' to exit): ")
    if input_val.lower() == 'q': break
    
    x_new = np.array([[float(input_val)]])
    y_pred = model.predict(x_new)
    print(f" Predicted value y: {y_pred[0]:.4f}")