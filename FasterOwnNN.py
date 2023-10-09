import numpy as np
import random


class Layer_Dense:
    def __init__(self, input_len, n_nodes, activation_function, layer_name):
        self.layer_name = layer_name
        self.n_nodes = n_nodes
        self.input_len = input_len
        self.activation = activation_function
        self.output = []
        self.weights = []
        self.bias = []

        # initialize layer with random weights
        for i in range(self.n_nodes):
            one_node_weights = np.array([random.random() * random.randint(-5, 5) for _ in range(input_len)])
            self.weights.append(one_node_weights)
            self.bias.append(random.random() * random.randint(-5, 5))
        self.weights = np.array(self.weights)

    def relu(self, val):
        if val < 0:
            return 0
        return val

    def sigmoid(self, val):
        if val >= 0:
            return 1 / (1 + np.exp(-val))
        else:
            return np.exp(val) / (1 + np.exp(val))


    def activation_function(self, value):
        if self.activation == 'relu':
            return self.relu(value)
        elif self.activation == 'sigmoid':
            return self.sigmoid(value)
        elif self.activation == 'linear':
            return value
        else:
            raise Exception("Bad activation function!")

    def forward(self,input):
        self.output = []
        for i,weight in enumerate(self.weights):
            node_val = np.matmul(weight,input)+self.bias[i]
            self.output.append(self.activation_function(node_val)) # matrix multiplication
        return self.output

    def random_chance_weights_alternation(self, chance, how_big):  # chance should be a number between 0 and 1
        self.output = []

        for i in range(len(self.weights)):
            if random.random() < chance:
                random_array = np.array([random.random() * random.choice([-1, 1]) * how_big for _ in range(self.input_len)])
                self.weights[i] = np.add(self.weights[i],random_array)
                self.bias[i] += random.random() * random.choice([-1, 1]) * how_big

    def print_weights(self):
        print(f'{self.layer_name}weights:{self.weights}')
