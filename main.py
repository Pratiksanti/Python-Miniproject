import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
hours = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
scores = np.array([12, 19, 33, 38, 52, 58, 69, 79])
model = LinearRegression()
model.fit(hours, scores)
try:
    user_input = input("Enter hours studied (comma-separated for multiple): ")
    user_hours = [float(h.strip()) for h in user_input.split(",")]
    input_array = np.array(user_hours).reshape(-1, 1)
    
    predicted_scores = model.predict(input_array)

  
    for h, s in zip(user_hours, predicted_scores):
        print(f"For {h} hours studied → Predicted Score: {s:.2f}")
#   graph from Here 
    plt.scatter(hours, scores, color='blue', label='Training Data')
    plt.plot(hours, model.predict(hours), color='green', label='Regression Line')
    plt.scatter(user_hours, predicted_scores, color='red', label='Your Predictions')
    plt.xlabel('Hours Studied')
    plt.ylabel('Exam Score')
    plt.title('Student Score Prediction using ML')
    plt.legend()
    plt.grid(True)
    plt.show()

except ValueError:
    print("Invalid input. Please enter numbers only.")
