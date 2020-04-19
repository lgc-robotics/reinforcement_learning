"""
Algorithm:
----------
Batch actor-critic algorithm. This code can be used to continuous action space game by
modifying function self.sample_action.

Reference:
----------
Tensorlayer: https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_AC.py
MorvanZhou's tutorial: https://morvanzhou.github.io/tutorials/
berkeleydeeprlcourse: https://github.com/berkeleydeeprlcourse/homework/blob/master/hw2/train_pg_f18.py

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
import numpy as np
import gym
import matplotlib.pyplot as plt

class ActorCritic:
    def __init__(self,action_num,n_layers,gamma=0.99,lr_actor=0.02,lr_critic=0.02,batch_size=1000,max_traj_length=500):
        self.action_num = action_num
        self.gamma = gamma
        self.min_timesteps_per_batch = batch_size
        self.max_traj_length = max_traj_length
        self.lr_actor=lr_actor
        self.lr_critic=lr_critic
        self.normalize_advantages=True
        self.is_animate=True

        self.episode_reward=[]

        self.actor = self.build_net(output_size=action_num, n_layers=n_layers)
        self.critic = self.build_net(output_size=1, n_layers=n_layers)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        #self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

    def build_net(self, output_size, n_layers, activation=tf.nn.relu, output_activation=None):
        kernel_init = 'he_normal'
        bias_init = tf.constant_initializer(0.01)
        model = tf.keras.Sequential()
        for i in range(len(n_layers)):
            model.add(Dense(n_layers[i], activation=activation, kernel_initializer=kernel_init, bias_initializer=bias_init))
        model.add(Dense(output_size, activation=output_activation, kernel_initializer=kernel_init,bias_initializer=bias_init))
        return model

    def sample_action(self, ob):
        # (state_dim,) => (batch, state_dim) and batch=1
        ob = ob[np.newaxis, :]
        ob = tf.convert_to_tensor(ob, dtype=tf.float32)
        # (batch, action_num)
        probs = self.actor(ob)
        # tf.random.categorical return a (batch,1) matrix
        a = tf.random.categorical(probs, 1)[0, 0]
        return a.numpy()

    def sample_trajectories(self,env,itr):
        timesteps_this_batch = 0
        trajs = []
        while True:
            animate_this_episode = (itr > 50 and self.is_animate)
            traj = self.sample_trajectory(env, animate_this_episode)
            trajs.append(traj)
            timesteps_this_batch += len(traj["reward"])
            if timesteps_this_batch >= self.min_timesteps_per_batch:
                break
        return trajs

    def sample_trajectory(self,env,animate_this_episode=False):
        ob_buffer, ac_buffer, rew_buffer = [], [], []
        ob = env.reset()
        steps = 0
        episode_reward = 0
        while True:
            if animate_this_episode:
                env.render()
            ac = self.sample_action(ob)
            ob_, rew, done, _ = env.step(ac)
            ob_buffer.append(ob)
            ac_buffer.append(ac)
            rew_buffer.append(rew)
            ob = ob_
            episode_reward += rew
            steps += 1
            if done or steps > self.max_traj_length:
                break
        print('Episode Reward:{:}'.format(episode_reward))
        self.episode_reward.append(episode_reward)
        traj = {'observation': np.array(ob_buffer),
                'reward': np.array(rew_buffer),
                'action': np.array(ac_buffer)}
        return traj

    def rewards_to_go(self,trajs):
        rewards_to_go=[]
        for traj in trajs:
            rew=traj["reward"]
            rew_to_go = np.zeros_like(rew)
            running_add = 0
            for t in reversed(range(0, len(rew))):
                running_add = running_add * self.gamma + rew[t]
                rew_to_go[t] = running_add
            rewards_to_go.append(rew_to_go)
        rewards_to_go=np.concatenate(rewards_to_go)
        return rewards_to_go

    def compute_advantage(self,obs,rewards_to_go):
        # (batch, state_dim)
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        # (batch, 1)
        baseline=self.critic(obs)
        # (batch, 1) => (batch, )
        baseline=tf.squeeze(baseline,axis=1)
        # (batch,)
        advantages=rewards_to_go-baseline.numpy()

        if self.normalize_advantages:
            advantages-=np.mean(advantages)
            advantages/=np.std(advantages)

        return advantages

    def fit_critic(self,obs,rewards_to_go):
        obs=tf.convert_to_tensor(obs,dtype=tf.float32)
        rewards_to_go=tf.convert_to_tensor(rewards_to_go,dtype=tf.float32)

        optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.lr_critic)
        # 50 is better than 20
        for i in range(50):
            with tf.GradientTape() as tape:
                # (batch, 1)
                v=self.critic(obs)
                # (batch,1) => (batch,)
                v=tf.squeeze(v)
                # (batch, )
                error=rewards_to_go-v
                loss=tf.reduce_mean(tf.math.square(error))
            grads=tape.gradient(loss,self.critic.trainable_variables)
            optimizer_critic.apply_gradients(zip(grads,self.critic.trainable_variables))

    def update_actor(self,obs,acs,rewards_to_go):
        advantages = self.compute_advantage(obs,rewards_to_go)

        with tf.GradientTape() as tape:
            logits = self.actor(obs)
            logits = tf.nn.softmax(logits)
            neg_log_prob = tf.reduce_sum(-tf.math.log(logits)*tf.one_hot(acs,depth=self.action_num),axis=1)
            loss = tf.reduce_mean(neg_log_prob * advantages)

        grads = tape.gradient(loss,self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grads,self.actor.trainable_variables))

    def train_actor_critic(self,env,seed,n_iter):
        # reproducible
        np.random.seed(seed)
        tf.random.set_seed(seed)
        env.seed(seed)

        all_episode_reward = []
        for itr in range(n_iter):
            print("**************Iteration %i**************"%itr)
            trajs = self.sample_trajectories(env,itr)

            # Build arrays for observation, action, rewards
            obs = np.concatenate([traj["observation"] for traj in trajs])
            acs = np.concatenate([traj["action"] for traj in trajs])
            rewards_to_go = self.rewards_to_go(trajs)

            # Update network
            self.fit_critic(obs,rewards_to_go)
            self.update_actor(obs,acs,rewards_to_go)

        env.close()

if __name__=='__main__':
    env = gym.make('CartPole-v1')
    agent = ActorCritic(action_num=env.action_space.n,
                        n_layers=[64],
                        lr_actor=0.01,
                        lr_critic=0.01,
                        max_traj_length=500,
                        batch_size=1000)

    agent.train_actor_critic(env, seed=116, n_iter=100)

    plt.figure()
    plt.plot(agent.episode_reward)
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.show()
