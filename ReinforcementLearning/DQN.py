import gym
import numpy as np

from keras.layers import *
from keras.models import Model

# Configurations

envName = "CartPole-v0"
env = gym.make(envName)

n_states  = env.observation_space.shape
n_actions = env.action_space.n

print("number of states:", n_states)
print("number of actions:", n_actions)


# Hyper parametres
memory_size = 10000
update_rate = 30
memroy_count = 0

batch_size = 64


n_episodes = 100

epsilon = 1
min_epsilon = 0.1
decay_rate = 0.001

gamma = 0.999 # Discount factor



class Agent():

    def __init__(self, n_states, epsilon,
                 min_epsilon, decay_rate,
                 memory_size, memory_count,
                 batch_size, update_rate, gamma):

        self.n_states = n_states[0]
        self.batch_size = batch_size
        self.q_valueModel = self.build_model(do_compile=True)
        self.target_qValueModel = self.build_model(do_compile=False)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.memory_size = memory_size
        self.memory = np.zeros(shape=(self.memory_size, 11))
        self.memory_count = memory_count
        self.gamma = gamma
        self.update_rate = update_rate
        self.train_counter = 0


    def build_model(self, do_compile):

        inputs = Input(shape=(self.n_states,))

        x = inputs

        x = Dense(units=256,
                  activation="relu",
                  kernel_initializer="he_normal")(x)
        x = Dense(units=256,
                  activation="relu",
                  kernel_initializer="he_normal")(x)
        x = Dense( units=256,
                   activation="relu",
                   kernel_initializer="he_normal" )(x)
        outputs = Dense(2,activation="linear")(x)

        model = Model(inputs, outputs)

        if do_compile:

            model.compile(loss="mse",
                          optimizer="Adam",
                          metrics=["accuracy"])
        return model

    def act(self, state):

        if self.epsilon < self.min_epsilon:

            random_index = np.argmax(self.q_valueModel.predict([[state]]))
            return random_index

        else:
            random_index = np.random.randint(low=0, high=1)
            self.epsilon = self.epsilon - self.decay_rate
            return random_index

    def remember(self, state, action, reward, next_state, done):

        self.memory[self.memory_count % self.memory_size] = np.array(list(state)+[action]+[reward]+list(next_state)+[done])
        self.memory_count +=1


    def do_train(self):

        self.train_counter +=1

        if self.memory_count > self.batch_size:
            random_indices = np.random.randint(0 ,min(self.memory_count, self.memory_size), size=self.batch_size)
            data = self.memory[random_indices]

            state = data[:, :self.n_states]
            action = data[:, self.n_states].astype("int")
            done = data[:, -1]
            next_state = data[:,self.n_states + 2:-1]
            reward = data[:, self.n_states + 1]
            x_train = state

            y = np.max(self.target_qValueModel.predict([next_state]), axis=1)
            y = reward + self.gamma*y*(1-done)

            y_train = self.q_valueModel.predict([state])

            y_train[np.arange(self.batch_size),action] = y

            if self.train_counter % self.update_rate == 0:
                self.target_qValueModel.set_weights(self.q_valueModel.get_weights())

            self.q_valueModel.train_on_batch(x_train, y_train)




agent = Agent( n_states, epsilon, min_epsilon,
               decay_rate,memory_size, memroy_count,
               batch_size, update_rate, gamma )


for episode in range(1, n_episodes + 1):

    state = env.reset()
    done = False
    totalRewards = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        totalRewards += reward
    agent.do_train()






