"""
ANN using numpy reference Code in Python

Reference :
https://gist.github.com/chmodsss/c445a433a4f87c6cbf4100630fb42168
https://towardsdatascience.com/neural-networks-from-scratch-easy-vs-hard-b26ddc2e89c7
Blog Title: Neural Networks from Scratch. Easy vs hard by : Sivasurya Santhanam

"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time


fileread_time_start = time.time()

dig = load_digits()
onehot_target = pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)

fileread_time_stop = time.time()
fileread_time = fileread_time_stop - fileread_time_start


"""
print(len(x_train))
print(len(x_val))
print(len(y_train))
print(len(y_val))
"""

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

class MyNN:
    def __init__(self, x, y):
        self.x = x
        self.neurons = 128
        self.lr = 0.5
        self.ip_dim = x.shape[1]
        self.op_dim = y.shape[1]

        self.w1 = np.random.randn(self.ip_dim, self.neurons)
        self.b1 = np.zeros((1, self.neurons))
        self.w2 = np.random.randn(self.neurons, self.neurons)
        self.b2 = np.zeros((1, self.neurons))
        self.w3 = np.random.randn(self.neurons, self.op_dim)
        self.b3 = np.zeros((1, self.op_dim))
        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)

    def backprop(self):
        loss = error(self.a3, self.y)
        print('Error in backprop :', loss)
        a3_delta = cross_entropy(self.a3, self.y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a3.argmax()

train_time_start = time.time()
model = MyNN(x_train, np.array(y_train))
epochs = 230
for x in range(epochs):
    print('training_epochs_iterations \t ', x, '\t', end="")
    model.feedforward()
    model.backprop()

def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100

#print("Training accuracy : ", get_acc(x_train, np.array(y_train)))

train_time_stop = time.time()
train_time = train_time_stop - train_time_start
print();
test_time_start = time.time()

testing_accuracy = get_acc(x_val, np.array(y_val))

test_time_stop = time.time()
test_time = test_time_stop - test_time_start

print("Testing accuracy : \t", testing_accuracy, "%")
print("file reading time : \t", fileread_time, " seconds")
print("Training time : \t", train_time, " seconds")
print("Testing time : \t\t", test_time, " seconds")
print();
print("num_training_Epocs : \t", epochs)
print("learning_rate : \t", model.lr)
print("num_layer1_Neurons : \t", model.ip_dim)
print("num_layer2_Neurons : \t", model.neurons)
print("num_layer3_Neurons : \t", model.neurons)
