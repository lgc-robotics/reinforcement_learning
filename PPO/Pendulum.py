import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input
import gym
import numpy as np
import matplotlib.pyplot as plt

ENV_ID = 'Pendulum-v0'

RANDOM_SEED = 1
RENDER = False

MAX_STEPS = 40000
HORIZON = 200
BATCH_SIZE = 32
EPOCHS = 10

GAMMA = 0.9
LR_A = 1e-4
LR_C = 2e-4
EPSILON = 0.2
ENTROPY_BETA=0.001
SAVE_FRE=10

class PPO(object):
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.actor = self.build_actor_net([100])
        self.critic = self.build_critic_net([100])

        self.actor_optimizer = tf.keras.optimizers.Adam(lr=LR_A)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=LR_C)

    def build_actor_net(self, n_layers,activation=tf.nn.relu):
        kernel_init = tf.random_uniform_initializer(0.1,1)
        inputs = Input(shape=[self.state_dim, ])
        for i in range(len(n_layers)):
            if i == 0:
                x = Dense(n_layers[i], activation=activation)(inputs)
            else:
                x = Dense(n_layers[i], activation=activation)(x)
        #mu=Dense(self.action_dim, activation=None, kernel_initializer=kernel_init)(x)
        mu = Dense(self.action_dim, activation=tf.nn.tanh)(x)
        mu = (mu + 1.) * (self.action_bound[1] - self.action_bound[0]) / 2. + self.action_bound[0]
        sigma = Dense(self.action_dim, activation=tf.nn.softplus, kernel_initializer=kernel_init)(x) #(0.1, 1) very important
        return tf.keras.Model(inputs=inputs, outputs=[mu, sigma])

    def build_critic_net(self, n_layers,activation=tf.nn.relu):
        #kernel_init = tf.random_uniform_initializer(0., 0.01)
        inputs = Input(shape=[self.state_dim, ])
        for i in range(len(n_layers)):
            if i == 0:
                x = Dense(n_layers[i], activation=activation)(inputs)
            else:
                x = Dense(n_layers[i], activation=activation)(x)
        outputs = Dense(1, activation=None)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


    @tf.function
    def cal_returns(self,rewards,states,last_state,done):
        V=self.critic(states)
        return1=rewards[:-1]+GAMMA*V[1:]
        if done:
            return2=rewards[-1:]
        else:
            return2=rewards[-1:]+GAMMA*self.critic(last_state[np.newaxis,:])
        returns=tf.concat([return1,return2],axis=0)
        
        advantages=returns-V
        advantages=tf.squeeze(advantages,axis=1)
        return returns,advantages
    
    @tf.function
    def sample_action(self, observation):
        observation = observation[np.newaxis, :]
        mu, sigma = self.actor(observation)
        pi = tfp.distributions.MultivariateNormalDiag(mu, sigma)  # (batch,)
        action = tf.clip_by_value(tf.squeeze(pi.sample(1), axis=0), self.action_bound[0], self.action_bound[1])
        return action[0]

    #@tf.function
    def update_actor(self, states, actions, advantages, old_pi):
        with tf.GradientTape() as tape:
            mean, std = self.actor(states)
            pi = tfp.distributions.MultivariateNormalDiag(mean, std)
            # return tensor shape = (batch,)
            ratio = pi.prob(actions) / (old_pi.prob(actions) + 1e-10)
            surr = tf.math.minimum(ratio * advantages, tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON) * advantages)
            entropy = pi.entropy()
            loss = -tf.math.reduce_mean(surr+entropy*ENTROPY_BETA)
            kl = tfp.distributions.kl_divergence(old_pi, pi)
            kl = tf.math.reduce_mean(kl)

        grad = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_weights))
        return kl

    #@tf.function
    def update_critic(self, states, n_step_returns):
        with tf.GradientTape() as tape:
            advantage = n_step_returns - self.critic(states)
            loss = tf.reduce_mean(0.5*tf.square(advantage))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.trainable_weights))

    # it should be commented to use 'kl'.
    #@tf.function
    def update(self, states, actions, returns, advantages):
        mean, std=self.actor(states)
        #old_pi=tfp.distributions.MultivariateNormalDiag(mean, std) #has problem!!!
        
        for i in range(EPOCHS):
            for idx in range(int(HORIZON/BATCH_SIZE)):
                bt_states=states[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                bt_actions=actions[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                bt_advantages=advantages[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                bt_returns=returns[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                
                bt_mean=mean[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                bt_std=std[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                bt_old_pi=tfp.distributions.MultivariateNormalDiag(bt_mean, bt_std)
                
                kl = self.update_actor(bt_states, bt_actions, bt_advantages, bt_old_pi)
                #if kl > tf.constant(1.5 * KL_TARGET):
                #    print('Early stopping at step %d due to reaching max kl.'%i)
                #    break
                
                self.update_critic(bt_states, bt_returns)
            
def train_ppo():
    #env = gym.make(ENV_ID).unwrapped
    env = gym.make(ENV_ID)
    
    # reproducible
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low, env.action_space.high]
    action_bound[0] = action_bound[0].reshape(1, action_dim)
    action_bound[1] = action_bound[1].reshape(1, action_dim)

    agent = PPO(state_dim, action_dim, action_bound)
    
    all_episode_reward = []
    total_steps=0
    episode_count=0
    while total_steps<MAX_STEPS:
        state_tensors,action_tensors,return_tensors,adv_tensors=[],[],[],[]
        state_buf, action_buf, reward_buf = [], [], []
        episode_reward = 0
        
        state = env.reset()
        state = state.astype(np.float32)
        for ii in range(HORIZON):  
            if RENDER:
                env.render()
            
            action = agent.sample_action(state) 
            action = action.numpy()
            state_, rew, done, _ = env.step(action)
            state_ = state_.astype(np.float32)

            state_buf.append(state)
            action_buf.append(action)
            reward_buf.append((rew+8)/10.)
            state = state_
            episode_reward += rew

            total_steps+=1

            if done or ii==HORIZON-1:
                # tensor shape: (B, dim)
                states = tf.convert_to_tensor(np.vstack(state_buf), dtype=tf.float32)
                actions = tf.convert_to_tensor(np.vstack(action_buf), dtype=tf.float32)
                rewards = tf.convert_to_tensor(np.vstack(reward_buf), dtype=tf.float32)
                returns, advs = agent.cal_returns(rewards,states,state_,done)

                state_tensors.append(states)
                action_tensors.append(actions)
                return_tensors.append(returns)
                adv_tensors.append(advs)
                
                state_buf, action_buf, reward_buf = [], [], []
                state = env.reset()
                state = state.astype(np.float32)
                
            if done:
                episode_count+=1
                print("Training | Episode:{}  | Steps: {}/{} | Episode Reward: {:.4f}".format(episode_count, total_steps, MAX_STEPS, episode_reward))
                if (episode_count+1)%SAVE_FRE==0:
                    agent.actor.save_weights('./checkpoints/actor_checkpoint'+str(episode_count))
                    agent.critic.save_weights('./checkpoints/critic_checkpoint'+str(episode_count))
                
                if len(all_episode_reward) < 5:  # record running episode reward
                    all_episode_reward.append(episode_reward)
                else:
                    all_episode_reward.append(episode_reward)
                    all_episode_reward[-1] = (np.mean(all_episode_reward[-5:]))  # smoothing
                    
                episode_reward = 0
                
        state_tensors=tf.concat(state_tensors,axis=0)
        action_tensors=tf.concat(action_tensors,axis=0)
        return_tensors=tf.concat(return_tensors,axis=0)
        adv_tensors=tf.concat(adv_tensors,axis=0)
        adv_tensors-=tf.math.reduce_mean(adv_tensors)
        adv_tensors/=(tf.math.reduce_std(adv_tensors)+1e-8)
        agent.update(state_tensors,action_tensors,return_tensors,adv_tensors)            
    
    plt.figure()
    plt.plot(all_episode_reward)
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.show()

def test_ppo():
    env = gym.make(ENV_ID).unwrapped    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low, env.action_space.high]
    action_bound[0] = action_bound[0].reshape(1, action_dim)
    action_bound[1] = action_bound[1].reshape(1, action_dim)

    agent = PPO(state_dim, action_dim, action_bound)
    
    agent.actor.load_weights('./checkpoints/actor_checkpoint')
    agent.critic.load_weights('./checkpoints/critic_checkpoint')

    while True:
        state = env.reset()
        state = state.astype(np.float32)
        
        print("Trying a new epoch...")

        for i in range(200):
            env.render()
            
            # shape of action, state : (1,) and (3,)
            action = agent.sample_action(state) 
            state_, rew, done, _ = env.step(action.numpy())
            state_ = state_.astype(np.float32)
            state = state_

            if done:
                break

if __name__ == '__main__':
    train_ppo()
