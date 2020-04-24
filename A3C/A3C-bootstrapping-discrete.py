"""
Algorithm:
----------
A3C algorithm. This code can be used to continuous action space game by
modifying function self.sample_action.

Reference:
----------
Tensorlayer: https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_AC.py
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

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense,Input
import numpy as np
import gym
import threading
import multiprocessing
import matplotlib.pyplot as plt

ENV_NAME = 'CartPole-v0'
N_LAYERS = [96]
LR_CRITIC = 1e-3
LR_ACTOR = 1e-3
N_STEPS = 10
GAMMA = 0.99
ENTROPY_BETA=0.05
CLIP_GRADIENT=50
MAX_GLOBAL_EPISODES = 1000
NUM_WORKERS = 4  # multiprocessing.cpu_count() can return the workers of your cpu
RENDER = False
SEED = 116

class ActorCritic(object):
    def __init__(self, state_dim, action_num):
        self.state_dim=state_dim
        self.action_num=action_num

        self.actor=self.build_net(N_LAYERS, self.action_num)
        self.critic=self.build_net(N_LAYERS,1)

    def build_net(self, n_layers, output_size):
        kernel_init = tf.random_uniform_initializer(0, 0.01)
        bias_init = tf.constant_initializer(0.01)
        inputs=Input(shape=[self.state_dim,])
        for i in range(len(n_layers)):
            if i==0:
                x=Dense(n_layers[i],activation=tf.nn.leaky_relu,kernel_initializer=kernel_init,bias_initializer=bias_init)(inputs)
            else:
                x=Dense(n_layers[i],activation=tf.nn.leaky_relu,kernel_initializer=kernel_init,bias_initializer=bias_init)(x)
        y=Dense(output_size,activation=None,kernel_initializer=kernel_init,bias_initializer=bias_init)(x)
        return tf.keras.Model(inputs=inputs,outputs=y)

    def sample_action(self,observation):
        # shape: (state_dim,) => (1, state_dim)
        observation = observation[np.newaxis, :]
        # shape: (1, action_num)
        logits = self.actor(observation)
        probs = tf.nn.softmax(logits)
        # np scalar
        action = np.random.choice(np.arange(self.action_num), p=probs.numpy()[0])
        return action

    def update_global(self,observations,actions,n_step_returns,master):
        """update critic"""
        with tf.GradientTape() as tape:
            # shape: (batch, 1)
            vs=self.critic(observations)
            advantages=n_step_returns-vs
            # tensor scalar
            critic_loss=tf.math.reduce_mean(tf.math.square(advantages))
        critic_grad=tape.gradient(critic_loss,self.critic.trainable_variables)
        #critic_grad = [tf.clip_by_value(grad, -CLIP_GRADIENT, CLIP_GRADIENT) for grad in critic_grad]
        master.share_critic_optimizer.apply_gradients(zip(critic_grad,master.share_ACnet.critic.trainable_variables))

        """update actor"""
        with tf.GradientTape() as tape:
            # shape (batch, action_num)
            logits = self.actor(observations)
            prob = tf.nn.softmax(logits)
            # shape (batch, 1)
            neg_log_prob = tf.math.reduce_sum(-tf.math.log(prob+1e-10) * tf.one_hot(actions, depth=self.action_num), axis=1)
            # entropy to encourage exploration, output shape: (1,1)
            entropy = -tf.math.reduce_mean(prob*tf.math.log(prob+1e-10))
            # loss
            actor_loss=tf.reduce_mean(neg_log_prob*advantages)+ENTROPY_BETA*entropy
        actor_grad=tape.gradient(actor_loss,self.actor.trainable_variables)
        #actor_grad=[tf.clip_by_value(grad,-CLIP_GRADIENT,CLIP_GRADIENT) for grad in actor_grad]
        master.share_actor_optimizer.apply_gradients(zip(actor_grad,master.share_ACnet.actor.trainable_variables))

    def pull_global(self,master):
        for l_p, g_p in zip(self.critic.trainable_weights, master.share_ACnet.critic.trainable_weights):
            l_p.assign(g_p)
        for l_p, g_p in zip(self.actor.trainable_weights, master.share_ACnet.actor.trainable_weights):
            l_p.assign(g_p)

class Worker(object):
    def __init__(self,work_name,state_dim,action_num):
        self.work_name=work_name
        self.env=gym.make(ENV_NAME)
        self.local_AC=ActorCritic(state_dim,action_num)

    def work(self,master):
        while not COORD.should_stop() and master.global_episodes<MAX_GLOBAL_EPISODES:
            observation_buffer, action_buffer, reward_buffer = [], [], []
            ob=self.env.reset()
            ob = ob.astype(np.float32)
            ep_r=0
            while True:
                if RENDER and self.work_name == 'Worker_0' and master.global_episodes>MAX_GLOBAL_EPISODES/2:
                    self.env.render()

                ac = self.local_AC.sample_action(ob)
                ob_, rew, done, _ = self.env.step(ac)
                ob_=ob_.astype(np.float32)

                ep_r += rew
                observation_buffer.append(ob)
                action_buffer.append(ac)
                reward_buffer.append(rew)

                if len(reward_buffer)==N_STEPS or done:
                    if done:
                        v_ = 0.
                    else:
                        v_ = self.local_AC.critic(ob_[np.newaxis,:])[0,0] # (1,1) => scalar
                        v_ = v_.numpy()

                    n_step_returns = []
                    for r in reward_buffer[::-1]:
                        v_ = r+ GAMMA * v_
                        n_step_returns.append(v_)
                    n_step_returns.reverse()

                    # list => Tensor (batch, dim)
                    observation_buffer=tf.convert_to_tensor(np.vstack(observation_buffer),dtype=tf.float32)
                    #action_buffer=tf.convert_to_tensor(np.vstack(action_buffer),dtype=tf.float32)
                    n_step_returns=tf.convert_to_tensor(np.vstack(n_step_returns),dtype=tf.float32)

                    # update global networks
                    self.local_AC.update_global(observation_buffer,action_buffer,n_step_returns,master)
                    observation_buffer,action_buffer,reward_buffer=[],[],[]

                    # update local network
                    self.local_AC.pull_global(master)
                ob = ob_
                if done:
                    print("Training | {}, Episode:{}/{} | Episode Reward: {:.1f}".format(
                        self.work_name,master.global_episodes,MAX_GLOBAL_EPISODES,ep_r))
                    master.global_episodes += 1
                    master.global_episode_rewards.append(ep_r)
                    break

class Master(object):
    def __init__(self,state_dim,action_num):
        self.share_ACnet=ActorCritic(state_dim=state_dim,action_num=action_num)
        self.share_actor_optimizer=tf.optimizers.Adam(learning_rate=LR_ACTOR)
        self.share_critic_optimizer=tf.optimizers.Adam(learning_rate=LR_CRITIC)
        self.global_episodes=0
        self.global_episode_rewards=[]

if __name__=="__main__":
    # reproducible
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    """get env parameters"""
    env=gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    env.close()

    """create a global actor critic"""
    master = Master(state_dim=state_dim,action_num=action_num)

    """Threading"""
    workers=[]
    for i in range(NUM_WORKERS):
        i_name='Worker_%i' % i
        workers.append(Worker(work_name=i_name,state_dim=state_dim,action_num=action_num))
    COORD=tf.train.Coordinator()
    workers_threads=[]
    for worker in workers:
        job = lambda : worker.work(master)
        t=threading.Thread(target=job)
        t.start()
        workers_threads.append(t)
    COORD.join(workers_threads)

    plt.figure()
    plt.plot(master.global_episode_rewards)
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.show()
