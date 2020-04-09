import gym
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import GradientTape
import matplotlib.pyplot as plt

plt.ion()


class PoligyGradientAgent:
    def __init__(self):
        self.STATE_SHAPE = (4,)
        self.N_ACTIONS = 2
        self.LR = 1e-4
        self.build_model()

    def build_model(self):
        states_input = Input(shape=self.STATE_SHAPE)
        fc1 = Dense(32, activation='relu')(states_input)
        fc2 = Dense(16, activation='relu')(fc1)
        probs = Dense(self.N_ACTIONS, activation='softmax')(fc2)
        self.model = Model(inputs=states_input, outputs=probs)

    def learn(self, states, actions_taken, rewards):
        for state, action, reward in zip(states, actions_taken, rewards):
            with GradientTape() as tape:
                probs = self.model(np.asarray([state]))
                J = categorical_crossentropy(np.asarray(action), probs) * -reward
            grads = tape.gradient(J, self.model.trainable_variables)
            for var_grad, var in zip(grads, self.model.trainable_variables):
                var.assign_add(var_grad * self.LR)

    def act(self, state):
        state_np = np.asarray([state])
        probs = self.model([state_np, np.asarray([1.]), np.asarray([1.])]).numpy()[0]
        action = np.random.choice([x for x in range(self.N_ACTIONS)], p=probs)
        return action


GAMMA = 0.99


def prep_rewards(rewards):
    q_values = [10] * len(rewards)
    q_values[-1] = -100
    for i in range(-2, -len(q_values) - 1, -1):
        q_values[i] = q_values[i + 1] * GAMMA
    return q_values


def normalize_returns(returns):
    returns = np.asarray(returns)
    mean = np.mean(returns)
    std = np.std(returns) if np.std(returns) > 1 else 1
    returns = (returns - mean) / std
    return returns


env = gym.make("CartPole-v1")
agent = PoligyGradientAgent()
scores = []
for train_iter in range(1000):
    train_iter_states_memory = []
    train_iter_actions_memory = []
    train_iter_returns_memory = []
    train_iter_scores_memory = []
    for game_no in range(32):
        game_states_memory = []
        game_rewards_memory = []
        game_actions_memory = []
        state = env.reset()
        done = False
        while not done:
            env.render()
            cur_action = agent.act(state)
            cur_action_onehot = np.zeros((agent.N_ACTIONS,))
            cur_action_onehot[cur_action] = 1
            game_actions_memory.append(cur_action_onehot)
            game_states_memory.append(state)
            state, cur_reward, done, _ = env.step(cur_action)
            game_rewards_memory.append(cur_reward)
        game_returns_memory = prep_rewards(game_rewards_memory)
        train_iter_scores_memory.append(len(game_returns_memory))
        [train_iter_states_memory.append(x) for x in game_states_memory]
        [train_iter_actions_memory.append(x) for x in game_actions_memory]
        [train_iter_returns_memory.append(x) for x in game_returns_memory]
    train_iter_returns_normal = normalize_returns(train_iter_returns_memory)
    scores.append(sum(train_iter_scores_memory) / len(train_iter_scores_memory))
    plt.plot(scores)
    plt.pause(0.01)
    agent.learn(train_iter_states_memory, train_iter_actions_memory, train_iter_returns_normal)
