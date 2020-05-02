"""
Reference:
----------
Tensorlayer: https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_A3C.py
MorvanZhou's tutorial: https://morvanzhou.github.io/tutorials/

Prerequisites
--------------
tensorflow 2.1
tensorflow-probability
gym
&&
pip install box2d box2d-kengz --user

Env
___
MountainCarContinuous-v0  (episode length: 1000)
BipedalWalker-v3
Pendulum-v0 (episode length: 200)
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense,Input
import numpy as np
import gym
import threading
import multiprocessing
import matplotlib.pyplot as plt

"""set hyper parameters"""
ENV_NAME = 'Pendulum-v0'
LR_CRITIC = 1e-3
LR_ACTOR = 1e-4
GAMMA = 0.90
ENTROPY_BETA = 0.01 # very bad to well: [0.1, 0.02, 0.005, 0.01]
N_STEPS = 5
MAX_LENGTH_PER_EPISODE = 200
MAX_GLOBAL_EPISODES = 2000
RENDER = False
SEED = 116
SHARED_OPTIMIZER = False
NUM_WORKERS = multiprocessing.cpu_count()

global_episodes=0
global_episode_rewards=[]

class ActorCritic(object):
    def __init__(self,state_dim,action_dim,action_bound):
        self.action_dim=action_dim
        self.state_dim=state_dim
        self.action_bound=action_bound

        self.actor=self.build_actor_net([200])
        self.critic=self.build_critic_net([100])

        self.actor_optimizer = tf.keras.optimizers.RMSprop(lr=LR_ACTOR)
        self.critic_optimizer=tf.keras.optimizers.RMSprop(lr=LR_CRITIC)

    def build_actor_net(self, n_layers):
        kernel_init = tf.random_normal_initializer(0., .1)
        bias_init = tf.constant_initializer(0.01)
        inputs=Input(shape=[self.state_dim,])
        for i in range(len(n_layers)):
            if i==0:
                x=Dense(n_layers[i],activation=tf.nn.relu6,kernel_initializer=kernel_init)(inputs)
            else:
                x=Dense(n_layers[i],activation=tf.nn.relu6,kernel_initializer=kernel_init)(x)
        mu=Dense(self.action_dim,activation=tf.nn.tanh,kernel_initializer=kernel_init)(x)
        mu = (mu+1.)*(self.action_bound[1]-self.action_bound[0])/2.+self.action_bound[0]
        sigma=Dense(self.action_dim,activation=tf.nn.softplus,kernel_initializer=kernel_init)(x)
        return tf.keras.Model(inputs=inputs,outputs=[mu,sigma])

    def build_critic_net(self,n_layers):
        kernel_init = tf.random_normal_initializer(0., .1)
        inputs = Input(shape=[self.state_dim,])
        for i in range(len(n_layers)):
            if i==0:
                x=Dense(n_layers[i],activation=tf.nn.relu6,kernel_initializer=kernel_init)(inputs)
            else:
                x=Dense(n_layers[i],activation=tf.nn.relu6,kernel_initializer=kernel_init)(x)
        outputs=Dense(1,activation=None,kernel_initializer=kernel_init)(x)
        return tf.keras.Model(inputs=inputs,outputs=outputs)

    # very important
    @tf.function
    def sample_action(self,observation):
        # shape: (state_dim,) => (batch, state_dim)
        observation = observation[np.newaxis, :]
        # shape: (batch, action_dim)
        mu,sigma = self.actor(observation)
        #dist=tfp.distributions.Normal(mu,sigma+1e-10) #(batch, action_dim)
        dist = tfp.distributions.MultivariateNormalDiag(mu, sigma + 1e-10) #(batch,)
        # shape: (1,action_dim)
        action=tf.clip_by_value(tf.squeeze(dist.sample(1),axis=0),self.action_bound[0],self.action_bound[1])
        return action[0]

    @tf.function
    def cal_critic_loss(self,observations,n_step_returns):
        vs = self.critic(observations)
        advantages = n_step_returns - vs
        critic_loss = tf.math.reduce_mean(tf.math.square(advantages))
        #if NORM_ADVAN:
        #    advantages-=tf.math.reduce_mean(advantages)
        #    advantages/=tf.math.reduce_std(advantages)
        return critic_loss,advantages

    @tf.function
    def cal_actor_loss(self,observations,actions,advantages):
        mu, sigma = self.actor(observations)
        #dist = tfp.distributions.Normal(mu, sigma + 1e-10)  # (batch, action_dim)
        dist = tfp.distributions.MultivariateNormalDiag(mu, sigma + 1e-10)  # (batch, 1)
        # shape (batch, )
        prob = dist.prob(actions)
        log_prob = tf.math.log(prob + 1e-10)
        # entropy to encourage exploration, output shape: (batch,)
        entropy = dist.entropy()
        # loss
        actor_loss = tf.reduce_mean(-log_prob * advantages - ENTROPY_BETA * entropy)
        return actor_loss

    def update_global(self,observations,actions,n_step_returns,master):
        # shapes of observations,actions,n_step_returns have been set to (batch, dim)
        """update critic"""
        with tf.GradientTape() as tape:
            critic_loss,advantages=self.cal_critic_loss(observations,n_step_returns)
        critic_grad=tape.gradient(critic_loss,self.critic.trainable_variables)

        """update actor"""
        with tf.GradientTape() as tape:
            actor_loss = self.cal_actor_loss(observations,actions,advantages)
        actor_grad=tape.gradient(actor_loss,self.actor.trainable_variables)

        if SHARED_OPTIMIZER:
            master.critic_optimizer.apply_gradients(zip(critic_grad, master.critic.trainable_variables))
            master.actor_optimizer.apply_gradients(zip(actor_grad, master.actor.trainable_variables))
        else:
            self.critic_optimizer.apply_gradients(zip(critic_grad, master.critic.trainable_variables))
            self.actor_optimizer.apply_gradients(zip(actor_grad,master.actor.trainable_variables))

    def pull_global(self,master):
        for l_p, g_p in zip(self.critic.trainable_weights, master.critic.trainable_weights):
            l_p.assign(g_p)
        for l_p, g_p in zip(self.actor.trainable_weights, master.actor.trainable_weights):
            l_p.assign(g_p)

class Worker(object):
    def __init__(self,work_name,state_dim,action_dim,action_bound):
        self.work_name=work_name
        self.env=gym.make(ENV_NAME)
        self.local_AC=ActorCritic(state_dim,action_dim,action_bound)

    def work(self,master):
        global global_episodes,global_episode_rewards
        while not COORD.should_stop() and global_episodes<MAX_GLOBAL_EPISODES:
            observation_buffer, action_buffer, reward_buffer = [], [], []
            ob=self.env.reset()
            ob = ob.astype(np.float32)
            ep_r=0
            steps_on_this_episode = 0
            while True:
                if RENDER and self.work_name == 'Worker_0':
                    self.env.render()

                ac = self.local_AC.sample_action(ob)
                ob_, rew, done, _ = self.env.step(ac)
                ob_=ob_.astype(np.float32)
                if steps_on_this_episode==MAX_LENGTH_PER_EPISODE-1:
                    done=True

                ep_r += rew
                observation_buffer.append(ob)
                action_buffer.append(ac)
                reward_buffer.append((rew+8)/8)

                ob = ob_
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
                    action_buffer=tf.convert_to_tensor(np.vstack(action_buffer),dtype=tf.float32)
                    n_step_returns=tf.convert_to_tensor(np.vstack(n_step_returns),dtype=tf.float32)

                    # update global networks
                    self.local_AC.update_global(observation_buffer,action_buffer,n_step_returns,master)

                    # update local network
                    self.local_AC.pull_global(master)

                    observation_buffer, action_buffer, reward_buffer = [], [], []

                if done:
                    if len(global_episode_rewards) < 5:  # record running episode reward
                        global_episode_rewards.append(ep_r)
                    else:
                        global_episode_rewards.append(ep_r)
                        global_episode_rewards[-1] = (np.mean(global_episode_rewards[-5:]))  # smoothing
                    print("Training | {}, Episode:{}/{} | Episode Reward: {:.4f}".format(
                        self.work_name,global_episodes,MAX_GLOBAL_EPISODES,ep_r))
                    global_episodes += 1
                    break

                steps_on_this_episode += 1

if __name__=="__main__":
    # reproducible
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    """get env parameters"""
    env=gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low, env.action_space.high]
    action_bound[0] = action_bound[0].reshape(1, action_dim)
    action_bound[1] = action_bound[1].reshape(1, action_dim)
    env.close()

    with tf.device("/cpu:0"):
        """create a global actor critic"""
        master = ActorCritic(state_dim=state_dim,action_dim=action_dim,action_bound=action_bound)
        """create local actor critic"""
        workers=[]
        for i in range(NUM_WORKERS):
            i_name='Worker_%i' % i
            workers.append(Worker(work_name=i_name,state_dim=state_dim,action_dim=action_dim,action_bound=action_bound))

    COORD=tf.train.Coordinator()
    workers_threads=[]
    for worker in workers:
        job = lambda : worker.work(master)
        t=threading.Thread(target=job)
        t.start()
        workers_threads.append(t)
    COORD.join(workers_threads)

    plt.figure()
    plt.plot(global_episode_rewards)
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.show()
