import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

dig = load_digits()
onehot_target = pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)
np.savetxt("xtrain.csv", x_train, delimiter=",")
np.savetxt("xval.csv", x_val, delimiter=",")
np.savetxt("ytrain.csv", y_train, delimiter=",")
np.savetxt("yval.csv", y_val, delimiter=",")

