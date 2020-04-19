"""
Algorithm:
----------
Vanilla Policy Gradient(REINFORCE). This code can be used to continuous action space game by
modifying function self.sample_action.

Reference:
----------
berkeleydeeprlcourse: https://github.com/berkeleydeeprlcourse/homework/blob/master/hw2/train_pg_f18.py
Tensorlayer: https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_PG.py
MorvanZhou's tutorial: https://morvanzhou.github.io/tutorials/

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

import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense

class PolicyGradient:
    def __init__(self,action_num,n_layers,size,activation=tf.nn.tanh,output_activation=None,
                 gamma=0.99,learning_rate=0.02,batch_size=1000,max_traj_length=500):
        self.action_num = action_num
        self.gamma = gamma
        self.min_timesteps_per_batch = batch_size
        self.max_traj_length = max_traj_length
        self.normalize_teward_to_go=True

        self.model=self._build_net(output_size=action_num,n_layers=n_layers,size=size,activation=activation,output_activation=output_activation)
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _build_net(self,output_size,n_layers,size,activation,output_activation):
        kernel_init = 'he_normal'
        bias_init = tf.constant_initializer(0.01)

        model = tf.keras.Sequential()

        for i in range(n_layers):
            model.add(Dense(size[i],activation=activation,kernel_initializer=kernel_init,bias_initializer=bias_init))
        model.add(Dense(output_size,activation=output_activation,kernel_initializer=kernel_init,bias_initializer=bias_init))

        return model

    def sample_action(self, state):
        # (state_dim,) => (batch, state_dim) and batch=1
        state = state[np.newaxis, :]
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        # (batch, action_num)
        logits = self.model(state)

        """
        # tf.random.categorical return a (batch,1) matrix
        a = tf.random.categorical(logits, 1)[0, 0]
        # Tensor to python scalar
        a = a.numpy()
        """

        probs = tf.nn.softmax(logits)
        a = np.random.choice(np.arange(self.action_num), p=probs.numpy()[0])
        return a

    def _sample_trajectories(self,env,itr):
        timesteps_this_batch = 0
        trajs = []
        while True:
            animate_this_episode = (itr > 30)
            traj = self._sample_trajectory(env,animate_this_episode)
            trajs.append(traj)
            timesteps_this_batch += len(traj["reward"])
            if timesteps_this_batch >= self.min_timesteps_per_batch:
                break
        return trajs

    def _sample_trajectory(self,env,animate_this_episode=False):
        ob_buffer, ac_buffer, rew_buffer = [], [], []
        ob=env.reset()
        steps=0
        episode_reward=0
        while True:
            if animate_this_episode:
                env.render()
            ac=self.sample_action(ob)
            ob_, rew, done, _ = env.step(ac)
            ob_buffer.append(ob)
            ac_buffer.append(ac)
            rew_buffer.append(rew)
            ob=ob_
            episode_reward+=rew
            steps+=1
            if done or steps > self.max_traj_length:
                break
        print('Episode Reward:{:}'.format(episode_reward))
        traj = {'observation':np.array(ob_buffer),
                'reward':np.array(rew_buffer),
                'action':np.array(ac_buffer)}
        return traj

    def _sum_of_rewards(self,trajs):
        rews=[]
        for traj in trajs:
            rew=traj["reward"]
            rew_to_go = np.zeros_like(rew)
            running_add = 0
            for t in reversed(range(0, len(rew))):
                running_add = running_add * self.gamma + rew[t]
                rew_to_go[t] = running_add
            rews.append(rew_to_go)
        rews=np.concatenate(rews)

        # an important trick: normalization could reduce variance
        if self.normalize_teward_to_go:
            rews-=np.mean(rews)
            rews/=np.std(rews)

        return rews

    def learn(self,obs,acs,rews,num_trajs):
        obs = tf.convert_to_tensor(obs,dtype=tf.float32)
        acs = tf.convert_to_tensor(acs,dtype=tf.int32)
        rews = tf.convert_to_tensor(rews,dtype=tf.float32)

        with tf.GradientTape() as tape:
            # (batch, action_num)
            logits = self.model(obs)
            logits = tf.nn.softmax(logits)

            # (batch,)
            neg_log_prob = tf.reduce_sum(-tf.math.log(logits)*tf.one_hot(acs,depth=self.action_num),axis=1)
            # (1,)
            loss = tf.reduce_sum(neg_log_prob * rews)/num_trajs

        grads = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))

    def train_policy_gradient(self,env,seed,n_iter):
        # reproducible
        np.random.seed(seed)
        tf.random.set_seed(seed)
        env.seed(seed)

        all_episode_reward = []
        for itr in range(n_iter):
            print("**************Iteration %i**************"%itr)
            trajs = self._sample_trajectories(env,itr)

            # Build arrays for observation, action, rewards
            obs = np.concatenate([traj["observation"] for traj in trajs])
            acs = np.concatenate([traj["action"] for traj in trajs])
            rews = self._sum_of_rewards(trajs)

            # Update network
            self.learn(obs,acs,rews,len(trajs))
        env.close()

if __name__=='__main__':
    env = gym.make('CartPole-v1')
    agent = PolicyGradient(action_num=env.action_space.n,
                           n_layers=1,
                           size=[64],
                           activation=tf.nn.tanh,
                           output_activation=None,
                           gamma=0.99,
                           learning_rate=0.02,
                           batch_size=1000,
                           max_traj_length=500)

    agent.train_policy_gradient(env, seed=116, n_iter=100)
