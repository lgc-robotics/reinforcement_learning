"""
Algorithm:
----------
Vanilla Policy Gradient(REINFORCE). We implement a simple version of REINFORCE, which does not estimate the
"baseline", and is only suitable to discrete action space game.

Reference:
----------
Tensorlayer: https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_PG.py
MorvanZhou's tutorial: https://morvanzhou.github.io/tutorials/

Environment:
------------
CartPole-v0
CartPole-v1

Prerequisites:
--------------
tensorflow 2
gym

"""

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt

class PolicyGradient:
    def __init__(self,action_num,state_dim,n_layers,size,gamma=0.99,learning_rate=0.02):
        self.action_num=action_num
        self.state_dim=state_dim

        self.gamma=gamma
        self.state_buffer,self.action_buffer,self.reward_buffer=[],[],[]

        self.model=self._build_net(action_num,n_layers,size)
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _build_net(self,output_size,n_layers,size,activation=tf.nn.tanh,output_activation=None):
        kernel_init='he_normal'
        bias_init=tf.constant_initializer(0.01)

        model=tf.keras.Sequential()

        for i in range(n_layers):
            model.add(layers.Dense(size[i],activation=activation,kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(layers.Dense(output_size,activation=output_activation,kernel_initializer=kernel_init,bias_initializer=bias_init))

        return model

    def store_transition(self,s,a,r):
        self.state_buffer.append(s[np.newaxis,:])
        self.action_buffer.append(a)
        self.reward_buffer.append(r)

    def select_action(self,state):
        # (state_dim,) => (batch, state_dim) and batch=1
        state=state[np.newaxis,:]
        state=tf.convert_to_tensor(state, dtype=tf.float32)

        # (batch, action_num)
        probs=self.model(state)

        # tf.random.categorical return a (batch,1) matrix
        a=tf.random.categorical(probs,1)[0,0]

        return a.numpy()

    def learn(self):
        with tf.GradientTape() as tape:
            # list of (state_dim,) => (batch, state_dim)
            states = np.vstack(self.state_buffer)
            states = tf.convert_to_tensor(states, dtype=tf.float32)

            # (batch,)
            reward_to_go = self._reward_to_go()

            # (batch, action_num)
            logits=self.model(states)
            logits=tf.nn.softmax(logits)

            # (batch,)
            neg_log_prob=tf.reduce_sum(-tf.math.log(logits)*tf.one_hot(self.action_buffer,depth=self.action_num),axis=1)

            loss=tf.reduce_mean(neg_log_prob * reward_to_go)

        grads=tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []

    def _reward_to_go(self):
        # discount episode rewards
        discounted_reward_buffer = np.zeros_like(self.reward_buffer)
        running_add = 0
        for t in reversed(range(0, len(self.reward_buffer))):
            running_add = running_add * self.gamma + self.reward_buffer[t]
            discounted_reward_buffer[t] = running_add

        # an important trick
        discounted_reward_buffer -= np.mean(discounted_reward_buffer)
        discounted_reward_buffer /= np.std(discounted_reward_buffer)

        return tf.constant(discounted_reward_buffer,dtype=tf.float32)

    def _baseline(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

def train_policy_gradient(env_name):
    env = gym.make(env_name)

    # reproducible
    seed = 116
    np.random.seed(seed)
    tf.random.set_seed(seed)
    env.seed(seed)

    agent = PolicyGradient(action_num=env.action_space.n,
                           state_dim=env.observation_space.shape[0],
                           n_layers=1,
                           size=[64],
                           gamma=0.99,
                           learning_rate=0.02)

    all_episode_reward = []
    for episode in range(100):
        episode_reward = 0.
        state = env.reset()
        while True:
            if True:
                env.render()

            action = agent.select_action(state)
            state_, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward)
            state = state_
            episode_reward += reward
            if done:
                break

        print('Training | Episode: {} | Episode Reward:{:}'.format(episode,episode_reward))
        agent.learn()
        all_episode_reward.append(episode_reward)

    env.close()

    plt.figure()
    plt.plot(all_episode_reward)
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.show()

if __name__=='__main__':
    train_policy_gradient('CartPole-v0')
