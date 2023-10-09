import tensorflow as tf
import numpy as np

# cost function = R(s) + max(a')Q(s',a') - Q(s,a)

class DeepQN:
    def __init__(self,state_size,output_shape,alpha):
        self.TAU = 1e-3  # Soft update parameter.
        self.output_shape = output_shape

        self.target_q_network = tf.keras.models.Sequential([ # NN used to create target examples
            tf.keras.layers.Input(shape=state_size, name='L0'),
            tf.keras.layers.Dense(units=64,activation='relu',name='L1'),
            tf.keras.layers.Dense(units=64, activation='relu', name='L2'),
            tf.keras.layers.Dense(units=output_shape, activation='linear', name='L3'),
        ])
        self.q_network = tf.keras.models.Sequential([ # NN that is updated every iteration
            tf.keras.layers.Input(shape=state_size, name='L0'),
            tf.keras.layers.Dense(units=64,activation='relu',name='L1'),
            tf.keras.layers.Dense(units=64, activation='relu', name='L2'),
            tf.keras.layers.Dense(units=output_shape, activation='linear', name='L3'),
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def forward(self,input):
        output = self.q_network.predict(input)
        return output

    def compute_loss(self,experiences, gamma, q_network, target_q_network):
        states,actions,rewards,next_states = experiences # unpack the data
        # problem with state shape
        loss = 0
        for i in range(len(experiences)):
            max_next_qsa = max(target_q_network(next_states[i])[0])
            y_target = rewards[i] + max_next_qsa * gamma

            q_values = max(q_network(states[i])[0]) # problem lies here ?
            loss += (y_target - q_values)**2 # and here ?
        print(loss)
        return loss

    def learn(self,training_data,gamma):
        # Calculate the loss
        with tf.GradientTape() as tape:
            loss = self.compute_loss(training_data, gamma, self.q_network, self.target_q_network)

        # Get the gradients of the loss with respect to the weights.
        gradients = tape.gradient(loss, self.q_network.trainable_variables)

        # Update the weights of the q_network.
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        # soft update the weights of target q_network
        for target_weights, q_net_weights in zip(
                self.target_q_network.weights, self.q_network.weights
        ):
            target_weights.assign(self.TAU * q_net_weights + (1.0 - self.TAU) * target_weights)

