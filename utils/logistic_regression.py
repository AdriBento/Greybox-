# imports
import matplotlib.pyplot as plt
import pandas as pd
from utils import make_x_y_data
import numpy as np
import pickle
from scipy.optimize import fmin_tnc

temperature = 1

train_path = '/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/train.txt'
val_path = '/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/val.txt'
test_path = '/home/abennetot/dataset/OD-MonuMAI/MonuMAI_dataset/test.txt'
# Parse data
train_images_path, train_x, train_y = make_x_y_data(train_path)
val_images_path, val_x, val_y = make_x_y_data(val_path)
test_images_path, test_x, test_y = make_x_y_data(test_path)

theta = np.zeros((train_x.shape[1], 1))

def sigmoid(x, temperature):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x, temperature))


def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x), temperature)

def cost_function(self, theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(self, theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

def fit(self, x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta,
                  fprime=gradient,args=(x, y.flatten()))
    return opt_weights[0]

parameters = fit(train_x, train_y, theta)

print(parameters)

def predict(self, x):
    theta = parameters[:, np.newaxis]
    return probability(theta, x)
def accuracy(self, x, actual_classes, probab_threshold=0.5):
    predicted_classes = (predict(x) >=
                         probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy * 100

accuracy(X, y.flatten())