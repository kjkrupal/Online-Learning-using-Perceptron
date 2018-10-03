import sys
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from numpy import linalg as la

def initialize_weights(shape):
    W = np.zeros((shape[1],))
    return W

def load_dataset(train_dataset_path, dev_dataset_path, test_dataset_path):
    
    train_dataset = pd.read_csv(train_dataset_path, header = None)
    dev_dataset = pd.read_csv(dev_dataset_path, header = None)
    test_dataset = pd.read_csv(test_dataset_path, header = None)
    
    concat_set = pd.concat([train_dataset, dev_dataset, test_dataset], keys=[0,1,2])
    
    temp = pd.get_dummies(concat_set, columns = [1,2,3,4,5,6,8])

    train, dev, test = temp.xs(0), temp.xs(1), temp.xs(2)

    y_train = train[[9]]
    y_dev = dev[[9]]
    y_test = test[[9]]

    y_train = y_train.values.reshape((y_train.shape[0]))
    y_dev = y_dev.values.reshape((y_dev.shape[0]))
    y_test = y_test.values.reshape((y_test.shape[0]))

    train = train.drop([9], axis = 1)
    dev = dev.drop([9], axis = 1)
    test = test.drop([9], axis = 1)
    
    X_train = train.iloc[:, :].values

    X_dev = dev.iloc[:, :].values

    X_test = test.iloc[:, :].values

    # Encode Y
    labelencoder_y = LabelEncoder()
    y_train = labelencoder_y.fit_transform(y_train)
    y_dev = labelencoder_y.fit_transform(y_dev)
    y_test = labelencoder_y.fit_transform(y_test)

    # Convert 0 to -1
    y_train[y_train == 0] = -1
    y_dev[y_dev == 0] = -1
    y_test[y_test == 0] = -1

    return (X_train, y_train, X_dev, y_dev, X_test, y_test)

def calculate_learning_rate(x, y, w):
    numerator = 1 - y * (np.dot(w, x))
    denominator = np.square(la.norm(x))
    learning_rate = numerator / denominator
    return learning_rate

def standard_perceptron(x_train, y_train, x_dev, y_dev, x_test, y_test, w, maxiter):
    iterations = []
    train_mistakes = []
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []

    for i in range(maxiter):
        mistakes = 0
        for t in range(x_train.shape[0]):
            y_hat = np.sign(np.dot(x_train[t], w))
            if y_hat == 0:
                y_hat = -1
            if y_hat != y_train[t]:
                mistakes = mistakes + 1
                w = w + y_train[t] * x_train[t]
        
        train_accuracy = calculate_accuracy(mistakes, x_train.shape[0])
        dev_accuracy = test_perceptron(x_dev, y_dev, w, 0)
        test_accuracy = test_perceptron(x_test, y_test, w, 0)
        
        train_mistakes.append(mistakes)
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)
        iterations.append(i + 1)
    
    return (train_accuracies, dev_accuracies, test_accuracies, train_mistakes, iterations)

def passive_aggressive_perceptron(x_train, y_train, x_dev, y_dev, x_test, y_test, w, maxiter):
    iterations = []
    train_mistakes = []
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []

    for i in range(maxiter):
        mistakes = 0
        for t in range(x_train.shape[0]):
            y_hat = np.sign(np.dot(x_train[t], w))
            if y_hat == 0:
                y_hat = -1
            if y_hat != y_train[t]:
                mistakes = mistakes + 1
                learning_rate = calculate_learning_rate(x_train[t], y_train[t], w)
                w = w + learning_rate * y_train[t] * x_train[t]
        
        train_accuracy = calculate_accuracy(mistakes, x_train.shape[0])
        dev_accuracy = test_perceptron(x_dev, y_dev, w, 0)
        test_accuracy = test_perceptron(x_test, y_test, w, 0)
        
        train_mistakes.append(mistakes)
        train_accuracies.append(train_accuracy)
        dev_accuracies.append(dev_accuracy)
        test_accuracies.append(test_accuracy)
        iterations.append(i + 1)

    return (train_accuracies, dev_accuracies, test_accuracies, train_mistakes, iterations)

def test_perceptron(x, y, w, b):
    mistakes = 0
    for t in range(x.shape[0]):
        y_hat = np.sign(np.dot(x[t], w) + b)
        if y_hat == 0:
            y_hat = -1
        if y_hat != y[t]:
            mistakes = mistakes + 1
    
    accuracy = calculate_accuracy(mistakes, x.shape[0])

    return accuracy

def calculate_accuracy(total_mistakes, total_examples):
    accuracy = 100 - (total_mistakes * 100 / total_examples)
    return accuracy

def plot_curves(x_axis, y_axis, title, x_lab, y_lab, filename):
    plt.plot(x_axis, y_axis, color = 'red', linestyle='solid', linewidth = 2)
    plt.title(title, loc = 'center')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.savefig(filename)
    plt.close(1)

def averaged_perceptron_smart(x, y, maxiter):
    iterations = []
    train_mistakes = []
    train_accuracies = []
    dev_accuracies = []
    test_accuracies = []
    total_mistakes = []
    w = np.zeros((x.shape[1],))
    u = np.zeros((x.shape[1],))
    b = 0
    beta = 0
    c = 1
    for i in range(maxiter):
        mistakes = 0
        for t in range(x.shape[0]):
            y_hat = np.sign(np.dot(x[t], w) + b)
            if y_hat == 0:
                y_hat = -1
            if y_hat != y[t]:
                mistakes = mistakes + 1
                w = w + y[t] * x[t]
                b = b + y[t]
                u = u + y[t] * x[t] * c
                beta = beta + y[t] * c
            c = c + 1
        total_mistakes.append(mistakes)
        iterations.append(i + 1)
    final_weights = w - (1/c) * u
    final_bias = b - (1/c) * beta
    return (final_weights, final_bias, total_mistakes, iterations)

def averaged_perceptron_naive(x, y, maxiter):
    w = np.zeros((x.shape[1],))
    w_sum = np.zeros((x.shape[1],))
    count = 0
    iterations = []
    total_mistakes = []
    for i in range(maxiter):
        mistakes = 0
        for t in range(x.shape[0]):
            y_hat = np.sign(np.dot(x[t], w))
            if y_hat == 0:
                y_hat = -1
            if y_hat != y[t]:
                mistakes = mistakes + 1
                w = w + y[t] * x[t]
                w_sum = w_sum + w
                count = count + 1
        total_mistakes.append(mistakes)
        iterations.append(i + 1)
    final_weights = w_sum/count
    return (final_weights, total_mistakes, iterations)

def general_learning_curve(X_train, y_train, X_dev, y_dev, X_test, y_test, W_train, maxiter):
    count = 0
    num_examples = []
    dev_accs_std = []
    test_accs_std = []
    dev_accs_pa = []
    test_accs_pa = []
    while count < 25000:
        count += 5000
        temp_train_x = X_train[:count,:]
        temp_train_y = y_train[:count]
        (train_acc_std, dev_acc_std, test_acc_std, train_mistakes_std, iterations) = standard_perceptron(temp_train_x, temp_train_y, X_dev, y_dev, X_test, y_test, W_train, maxiter)
        (train_acc_pa, dev_acc_pa, test_acc_pa, train_mistakes_pa, iterations) = passive_aggressive_perceptron(temp_train_x, temp_train_y, X_dev, y_dev, X_test, y_test, W_train, maxiter)
        dev_accs_std.append(dev_acc_std[-1])
        test_accs_std.append(test_acc_std[-1])
        dev_accs_pa.append(dev_acc_pa[-1])
        test_accs_pa.append(test_acc_pa[-1])
        num_examples.append(count)
    
    return (dev_accs_std, test_accs_std, dev_accs_pa, test_accs_pa, num_examples)

if __name__ == '__main__':
    training_set = sys.argv[1]
    development_set = sys.argv[2]
    test_set = sys.argv[3]
    
    maxiter = 20
    
    (X_train, y_train, X_dev, y_dev, X_test, y_test) = load_dataset(training_set, development_set, test_set)
    W_train = initialize_weights(X_train.shape)

    # For Standard Perceptron 
    (train_acc_std, dev_acc_std, test_acc_std, train_mistakes_std, iterations) = standard_perceptron(X_train, y_train, X_dev, y_dev, X_test, y_test, W_train, maxiter)
    
    plot_curves(iterations, train_mistakes_std, 'Learning Curve (Standard Perceptron)', 'Iterations', 'Mistakes', 'std_learning_curve.png')
    plot_curves(iterations, train_acc_std, 'Training Accuracy Curve (Standard Perceptron)', 'Iterations', 'Accuracy', 'std_train_acc.png')
    plot_curves(iterations, dev_acc_std, 'Validation Accuracy Curve (Standard Perceptron)', 'Iterations', 'Accuracy', 'std_dev_acc.png')
    plot_curves(iterations, test_acc_std, 'Testing Accuracy Curve (Standard Perceptron)', 'Iterations', 'Accuracy', 'std_test_acc.png')
    
    # For PA update Perceptron 
    (train_acc_pa, dev_acc_pa, test_acc_pa, train_mistakes_pa, iterations) = passive_aggressive_perceptron(X_train, y_train, X_dev, y_dev, X_test, y_test, W_train, maxiter)
    
    plot_curves(iterations, train_mistakes_pa, 'Learning Curve (PA Update Perceptron)', 'Iterations', 'Mistakes', 'pa_learning_curve.png')
    plot_curves(iterations, train_acc_pa, 'Training Accuracy Curve (PA Update Perceptron)', 'Iterations', 'Accuracy', 'pa_train_acc.png')
    plot_curves(iterations, dev_acc_pa, 'Validation Accuracy Curve (PA Update Perceptron)', 'Iterations', 'Accuracy', 'pa_dev_acc.png')
    plot_curves(iterations, test_acc_pa, 'Testing Accuracy Curve (PA Update Perceptron)', 'Iterations', 'Accuracy', 'pa_test_acc.png')
    
    (weights_naive, mistakes_naive, iterations_naive) = averaged_perceptron_naive(X_train, y_train, maxiter)
    train_accuracy_naive = calculate_accuracy(mistakes_naive[-1], X_train.shape[0])
    dev_accuracy_naive = test_perceptron(X_dev, y_dev, weights_naive, 0)
    test_accuracy_naive = test_perceptron(X_test, y_test, weights_naive, 0)


    (weights_smart, bias, mistakes_smart, iterations_smart) = averaged_perceptron_smart(X_train, y_train, maxiter)
    train_accuracy_smart = calculate_accuracy(mistakes_smart[-1], X_train.shape[0])
    dev_accuracy_smart = test_perceptron(X_dev, y_dev, weights_smart, bias)
    test_accuracy_smart = test_perceptron(X_test, y_test, weights_smart, bias)

    print("Naive Training: ", train_accuracy_naive)
    print("Naive Validation: ", dev_accuracy_naive)
    print("Naive Testing: ", test_accuracy_naive)
    
    print("Smart Training: ", train_accuracy_smart)
    print("Smart Validation: ", dev_accuracy_smart)
    print("Smart Testing: ", test_accuracy_smart)

    (dev_accs_std, test_accs_std, dev_accs_pa, test_accs_pa, num_examples) = general_learning_curve(X_train, y_train, X_dev, y_dev, X_test, y_test, W_train, maxiter)
    plot_curves(num_examples, dev_accs_std, 'General Learning Curve for Validation (Standard Perceptron)', 'Number of examples', 'Accuracy', 'std_general_dev.png')
    plot_curves(num_examples, test_accs_std, 'General Learning Curve for Test (Standard Perceptron)', 'Number of examples', 'Accuracy', 'std_general_test.png')
    plot_curves(num_examples, dev_accs_pa, 'General Learning Curve for Validation (PA Update Perceptron)', 'Number of examples', 'Accuracy', 'pa_general_dev.png')
    plot_curves(num_examples, test_accs_pa, 'General Learning Curve for Test (PA Update Perceptron)', 'Number of examples', 'Accuracy', 'pa_general_test.png')

    