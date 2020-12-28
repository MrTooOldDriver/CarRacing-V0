import copy

import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

#random action
#5 try eps pick random action
#pick highest reward

#action space 5
#nothing
#brake
#acc
#left
#right

#put into replay buffer

def action_space():
    return {
        0 : [0,0,0],
        1 : [0,0,1],
        2 : [0,1,0],
        3 : [-1,0,0],
        4 : [1,0,0]
    }


def initial_network():
    initializer = tf.keras.initializers.LecunUniform()
    model = keras.Sequential()
    model.add(keras.Input(shape=(1,7056)))
    # model.add(tf.keras.layers.Conv2D(1,4, input_shape=(96,96,3)))
    model.add(layers.Flatten())
    # model.add(layers.Dense(576,activation="relu"))
    # model.add(layers.Dense(576,activation="relu"))
    # model.add(layers.Dense(144,activation="relu"))
    model.add(layers.Dense(512,kernel_initializer=initializer,activation="relu"))
    model.add(layers.Dense(5, kernel_initializer=initializer,activation="linear"))
    model.compile(loss='mse', optimizer="Adamax")
    model.summary()
    return model

def process_image(image_array):
    #cropping image to 84*84. remove the information bar. Only the track view
    crop_image = image_array[:84,6:90]
    tensor = tf.convert_to_tensor(crop_image)
    # print("tensor")
    grayscale = tf.image.rgb_to_grayscale(tensor)
    #print(grayscale)
    # print(tf.reshape(grayscale, [1, -1]))

    return tf.reshape(grayscale, [1, -1])

class QNetwork:
    def __init__(self):
        self.network = initial_network()

    def train(self,state,expect):
        # print(state)
        processed_state = []
        for s in state:
            processed_state.append(process_image(s))
        self.network.fit(np.array(processed_state),np.array(expect))

    def predict(self,state):
        processed_state = process_image(state)
        return self.network.predict(processed_state)

    def action(self,state,epsilon):
        processed_state = process_image(state)
        q = self.network.predict(processed_state)
        if np.random.random() < epsilon:
            return np.random.randint(0,4),q
        else:
            return np.argmax(q),q

def play(environment,epsilon,replay_buffer):
    return

def collect_data(environment, epsilon,replay_buffer, num_plays, network, gamma):
    # for _ in num_plays:
    observation = environment.reset()
    iteration_count = 0
    total_reward = 0
    short_term_reward = 0
    short_term_observation_buffer = []
    short_term_target_buffer = []
    short_term_action_index_buffer = []
    while True:
        environment.render()
        action_index, network_results = network.action(observation,epsilon)
        next_observation, reward, done, info = environment.step(action_space()[action_index])

        if iteration_count == 0:
            reward = -1.0
        # if reward > 0:
        #     print("chosing action{0}. reward={1}".format(action_space()[action_index],reward*10))

        next_results = network.predict(next_observation)
        # print(network_results)

        # network_target = network_results[:]
        # network_target[0][action_index] = reward*10 + gamma*np.max(next_results)
        # network.train(observation,network_target)#learn while in episode

        network_target = network_results[:]
        network_target[0][action_index] = reward * 10 + gamma * np.max(next_results)
        short_term_observation_buffer.append(observation)
        short_term_action_index_buffer.append(action_index)
        short_term_target_buffer.append(network_target)

        observation = next_observation[:]
        iteration_count +=1
        total_reward += reward
        short_term_reward += reward

        if iteration_count % 10 == 0:
            print(short_term_reward)
            if short_term_reward > 0: #if in last 10 steps there is a great move
                print("reward add up. short_term_reward={0}.".format(short_term_reward))
                reward_bias = 10
                for i in range(len(short_term_action_index_buffer)):
                    index = short_term_action_index_buffer[i]
                    short_term_target_buffer[i][0][index] += short_term_reward
            network.train(short_term_observation_buffer,short_term_target_buffer)
            short_term_reward = 0
            short_term_observation_buffer = []
            short_term_target_buffer = []
            short_term_action_index_buffer = []


        if done:
            print("episode done")
            break
        if iteration_count >2000:
            print("episode iteration limit reach")
            break
        if total_reward < -10:
            print("total_reward limit reach")
            break
    return iteration_count,total_reward
    # return

def main():
    env = gym.make('CarRacing-v0')
    gamma = 0.99

    num_of_plays = 100
    network = QNetwork()

    for n in range(num_of_plays):
        epsilon = 0.3 - 0.005*n
        if epsilon <= 0:
            epsilon = 0.02
        iteration_count,total_reward = collect_data(env,epsilon,None,None,network,gamma)
        print("Play{0} epsilon={1} iteration_count={2} total_reward={3} average_reward={4}".
              format(n,epsilon,iteration_count,total_reward,total_reward/iteration_count))
    return

main()


# env = gym.make('CarRacing-v0')
# # print(env.action_space)
# # print(env.action_space.high)
# # print(env.action_space.low)
# print(env.observation_space)
#
# obs = env.reset()
# for _ in range(1000):
#     env.render()
#     # print(env.action_space.sample())
#     # env.step([0,0.1,0.05]) # take a random action
# env.close()