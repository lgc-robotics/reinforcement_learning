"""
Algorithm:
----------
Online actor-critic algorithm. This code can be used to continuous action space game by
modifying function self.sample_action.

Environment:
------------
CartPole-v0
CartPole-v1
#MountainCar-v0

Prerequisites:
--------------
tensorflow 2
gym

"""

import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import math
import gym

class ActorCritic:
    def __init__(self,action_num,n_layers,gamma=0.99,lr_actor=0.02,lr_critic=0.02,n_step_returns=1,isRender=False):
        self.render=isRender
        self.action_num=action_num
        self.n_steps=n_step_returns
        self.gamma=gamma

        self.optimizer_actor=tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.optimizer_critic=tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.actor=self.build_net(action_num,n_layers)
        self.critic=self.build_net(1,n_layers)

    def build_net(self, output_size, n_layers, activation=tf.nn.leaky_relu, output_activation=None, kernel_regularizer=None):
        kernel_init = tf.random_uniform_initializer(0, 0.01)
        bias_init = tf.constant_initializer(0.01)
        model = tf.keras.Sequential()
        for i in range(len(n_layers)):
            model.add(Dense(n_layers[i], activation=activation, kernel_initializer=kernel_init, bias_initializer=bias_init,kernel_regularizer=kernel_regularizer))
        model.add(Dense(output_size, activation=output_activation, kernel_initializer=kernel_init,bias_initializer=bias_init,kernel_regularizer=kernel_regularizer))
        return model

    def sample_action(self, observation):
        # shape: (state_dim,) => (batch, state_dim) and batch=1
        observation = observation[np.newaxis, :]
        # shape: (batch, action_num)
        logits = self.actor(observation)
        """
        # tf.random.categorical return a (batch,1) matrix
        a = tf.random.categorical(logits, 1)[0, 0]
        # Tensor to python scalar
        a=a.numpy()
        """
        probs = tf.nn.softmax(logits)
        a = np.random.choice(np.arange(self.action_num), p=probs.numpy()[0])

        # try exploration
        #if np.random.rand()<0.2:
        #    a=np.random.randint(self.action_num,size=1)[0]
        return a

    def n_step_return(self, obs, rews, termimals):
        if termimals[-1]:
            v_t_n=tf.constant([0.],dtype=tf.float32)
        else:
            # shape: (1,1)=>(1,)
            v_t_n = self.critic(obs[-1][np.newaxis,:]) # value of state at time step (t+n)
            v_t_n = tf.squeeze(v_t_n, axis=0)

        # shape: (1,)
        returns = math.pow(self.gamma, self.n_steps) * v_t_n.numpy()
        for i in range(self.n_steps):
            returns += math.pow(self.gamma, i) * rews[i]
        return returns

    def n_step_advantage(self, obs, rews, termimals):
        # shape: (1,1)=>(1,)
        v_t=self.critic(obs[0][np.newaxis,:]) # value of state at time step (t)
        v_t=tf.squeeze(v_t,axis=0)
        # shape: (1, )
        n_step_return=self.n_step_return(obs,rews,termimals)
        # shape: (1, )
        advantage=n_step_return-v_t.numpy()
        return advantage

    def update_critic(self, obs, rews, termimals):
        # shape: (1,) => (1,1)
        n_step_rewards = self.n_step_return(obs, rews, termimals)
        n_step_rewards = tf.constant([n_step_rewards])

        with tf.GradientTape() as tape:
            # shape: (1,1)
            v_t = self.critic(obs[0][np.newaxis, :])  # value of state at time step (t)
            # shape: (1,1)
            td_error=n_step_rewards-v_t
            # Tensor scalar
            loss=tf.math.reduce_mean(tf.math.square(td_error))
        grad=tape.gradient(loss,self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(grad,self.critic.trainable_variables))

    def update_actor(self, obs, acs, rews, termimals):
        # shape: (1,)
        advantage = self.n_step_advantage(obs, rews, termimals)

        with tf.GradientTape() as tape:
            # shape: (1,1)
            observation = obs[0][np.newaxis, :]
            # shape: (1,1)
            logits = self.actor(observation)
            # shape: (1,)
            prob = tf.nn.softmax(logits)
            neg_log_prob = tf.math.reduce_sum(-tf.math.log(prob) * tf.one_hot([acs[0]], depth=self.action_num), axis=1)
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[acs[0]], logits=logits)
            # Tensor scalar
            loss = tf.math.reduce_mean(neg_log_prob * advantage)
        grads=tape.gradient(loss,self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grads,self.actor.trainable_variables))

    def train_actor_critic(self,env,seed,episodes):
        # reproducible
        np.random.seed(seed)
        tf.random.set_seed(seed)
        env.seed(seed)

        all_episode_reward = []
        for episode in range(episodes):
            observation_buffer, action_buffer, reward_buffer, termimal_buffer = [], [], [], []
            episode_reward = 0
            observation = env.reset().astype(np.float32)
            observation_buffer.append(observation)
            while True:
                if self.render and episode > episodes/2:
                    env.render()

                action = self.sample_action(observation)
                observation_, reward, done, _ = env.step(action)
                observation_ = observation_.astype(np.float32)

                observation_buffer.append(observation_)
                action_buffer.append(action)
                reward_buffer.append(reward)
                termimal_buffer.append(done)
                episode_reward += reward

                observation = observation_
                if len(action_buffer) == self.n_steps:
                    #print("updating...")
                    self.update_critic(observation_buffer,reward_buffer,termimal_buffer)
                    self.update_actor(observation_buffer,action_buffer,reward_buffer,termimal_buffer)

                    observation_buffer.pop(0)
                    action_buffer.pop(0)
                    reward_buffer.pop(0)
                    termimal_buffer.pop(0)
                if done:
                    break

            print('Training | Episode: {} | Episode Reward:{:}'.format(episode, episode_reward))
            all_episode_reward.append(episode_reward)

        plt.figure()
        plt.plot(all_episode_reward)
        plt.xlabel('episodes')
        plt.ylabel('total reward per episode')
        plt.show()

if __name__=='__main__':
    env = gym.make('CartPole-v1')
    agent = ActorCritic(action_num=env.action_space.n,
                        n_layers=[64],
                        gamma=0.99,
                        lr_actor=1e-3,
                        lr_critic=1e-2,
                        n_step_returns=2,
                        isRender=False)
    agent.train_actor_critic(env, seed=1, episodes=500)
    env.close()
