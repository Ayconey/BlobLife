import numpy as np
import random


class Node:
    def __init__(self, input_len):
        self.weights = np.array([random.random() * random.randint(-5, 5) for _ in range(input_len)])
        self.bias = np.random.random() * random.randint(-5, 5)

    def value(self,input):
        return np.dot(input, self.weights) + self.bias

    def modify_weights_bias_random(self, how_big):
        for i in range(len(self.weights)):
            self.weights[i] += random.random() * how_big
        self.bias += random.random() * how_big


class Layer_Dense:
    def __init__(self, input_len, n_nodes, activation_function, layer_name):
        self.layer_name = layer_name
        self.n_nodes = n_nodes
        self.input_len = input_len
        self.activation = activation_function
        self.nodes = []
        self.output = []
        self.weights = []
        self.bias = []

        # initialize layer
        for i in range(self.n_nodes):
            node = Node(self.input_len)
            self.nodes.append(node)
            self.weights.append(node.weights)
            self.bias.append(node.bias)

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
        for node in self.nodes:
            self.output.append(self.activation_function(node.value(input)))

        return self.output

    def random_chance_weights_alternation(self, chance, how_big):  # chance should be a number between 0 and 1
        self.weights = []
        self.bias = []
        self.output = []

        for node in self.nodes:
            if random.random() < chance:
                node.modify_weights_bias_random(how_big)
            self.weights.append(node.weights)
            self.bias.append(node.bias)

    def print_weights(self):
        print(f'{self.layer_name}weights:{self.weights}')
