import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Read the Excel file
df = pd.read_csv('Data/train.csv')

# Assuming the columns you want to use are named 'feature1', 'feature2', and 'feature3'
# Replace these names with the actual column names from your Excel file
features = ['Pclass', 'Sex', 'Age']
target = 'Survived'


# Handle categorical data by encoding
# Example: If 'Sex' is categorical
# 1 = Male and 0 = Female
if df['Sex'].dtype == 'object':
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

# Select the columns and convert to NumPy array
X_train = df[features].to_numpy()
y_train = df[target].to_numpy()

# Apply StandardScaler
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_train)



def plot_data():
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points with target == 1 as 'x' and target == 0 as 'o'
    # Sequence is ax.scatter (x,y,z)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], c='r', marker='x', label='Yes')
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], c='b', marker='o', label='No')

    # Labels
    ax.set_xlabel('Social Class')
    ax.set_ylabel('Sex')
    ax.set_zlabel('Age')

    # Add legend
    ax.legend()

    # Show plot
    plt.show()


def sigmoid(z):
    # Calculate the sigmoid function to lay all the prediction between 0 and 1
    g = 1/(1+np.exp(-z))

    return g

def compute_cost_logistic(X,y,w,b):
    # Define number of training examples
    m = X.shape[0]
    cost=0.0
    epsilon = 1e-10  # Small constant to prevent log(0)

    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i+epsilon) - (1-y[i])*np.log(1-f_wb_i+epsilon)
    
    cost = cost/m
    return cost

def compute_gradient_logistic(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w)+b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        dj_db = dj_db + err_i
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_db, dj_dw




# dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_train,y_train,w_tmp,b_tmp)

# print (f"dj_db: {dj_db_tmp}")
# print (f"dj_dw: {dj_dw_tmp}")


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    # Array of all costs that was obtained through iterations
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Updated weight and bias simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration

        if i < 100000:    # this is just to prevent resource exaustion
            J_history.append(compute_cost_logistic(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print (f"Iteration {i:4d}: Cost {J_history[-1]}")


    return w, b, J_history     # return final w, b, J history


# w_tmp = np.zeros_like(X_train[0])
# b_tmp = 0.
# alph = 0.01
# iters = 200000

# w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)

# print (f"\nUpdated Parameters: w:{w_out}, b:{b_out}")


# Updated Parameters: w:[-1.16773048 -2.6118988  -0.03342503], b:4.732206526674033

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

def accuracy(y_true, y_pred):
    # Ensure that y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)
    
    # Calculate accuracy as a percentage
    accuracy = (correct_predictions / len(y_true)) * 100
    
    return accuracy

w = np.array([-1.16773048,-2.6118988,-0.03342503])
b = 4.732206526674033
m = X_train.shape[0]

pred_array = predict(X_train, w, b)


# !!! Calculate the prediction accuracy

for i in range(m):
    if (pred_array[i] == y_train[i]):
        print(f"Match: {pred_array[i]}")
    else:
        print(f"No match: Predicted: {pred_array[i]} Actual: {y_train[i]}")


print (f"Accuracy: {accuracy(y_train,pred_array)}")