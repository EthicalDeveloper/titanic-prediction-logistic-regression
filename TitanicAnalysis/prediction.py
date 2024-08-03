import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Read the Excel file
df = pd.read_csv('Data/test.csv')

# Assuming the columns you want to use are named 'feature1', 'feature2', and 'feature3'
# Replace these names with the actual column names from your Excel file
features = ['Pclass', 'Sex', 'Age']

# Handle categorical data by encoding
# Example: If 'Sex' is categorical
# 1 = Male and 0 = Female
if df['Sex'].dtype == 'object':
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

# Select the columns and convert to NumPy array
X = df[features].to_numpy()


def sigmoid(z):
    # Calculate the sigmoid function to lay all the prediction between 0 and 1
    g = 1/(1+np.exp(-z))

    return g

def predict(X, w, b):
    m = X.shape[0]  # Number of training examples
    prediction = np.zeros(m)  # Initialize prediction array with zeros

    for i in range(m):
        z_i = np.dot(X[i], w) + b  # Compute the linear combination
        f_wb_i = sigmoid(z_i)  # Apply the sigmoid function
        if f_wb_i >= 0.5:
            prediction[i] = 1  # Assign 1 if probability is >= 0.5
        else:
            prediction[i] = 0  # Assign 0 if probability is < 0.5

    return prediction


w = np.array([-1.16773048,-2.6118988,-0.03342503])
b = 4.732206526674033
m = X.shape[0]

pred_array = predict(X, w, b)


for i in range(m):
    if pred_array[i] == 0 and X[i][1] == 0:
        print(f"Social Status: {X[i][0]} | Gender: {X[i][1]} | Age: {X[i][2]} | Prediction: Does Not Survive")
    if pred_array[i] == 1 and X[i][1] == 0:
        print(f"Social Status: {X[i][0]} | Gender: {X[i][1]} | Age: {X[i][2]} | Prediction: Survives")