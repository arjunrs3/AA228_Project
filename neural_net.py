from sklearn.neural_network import MLPRegressor
import tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Model Definition 
class NeuralNet(): 
    def __init__(self, n_layers, hidden_neurons, max_iter):
        self.model = MLPRegressor(hidden_layer_sizes=np.full(n_layers, hidden_neurons), 
                                  activation = "relu", 
                                  max_iter = max_iter, 
                                  )

    def train(self, X_train, y_train): 
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

if __name__ == "__main__": 
    with open("outputs", "rb") as f: 
        outputs = pickle.load(f)

    with open("inputs", "rb") as f: 
        inputs = pickle.load(f)

    df = pd.DataFrame(np.c_[inputs, outputs]).dropna()
    outputs = df.iloc[:, -2:]
    inputs = df.iloc[:, :-2]

    Q = NeuralNet(2, 64, 1000) 
    Q.train(inputs, outputs)
    with open("NN", "wb") as f: 
        pickle.dump(Q, f)
    pd.DataFrame(Q.model.loss_curve_).plot()
    plt.show()