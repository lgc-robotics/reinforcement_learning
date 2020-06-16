"""
reference:
https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ddpg/ddpg.py
https://github.com/tensorlayer/tensorlayer/blob/master/examples/reinforcement_learning/tutorial_DDPG.py
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input
from buffer import ReplayBuffer

class DDPG():
    def __init__(self,env,h_layers=[64,64],seed=0,epochs=100,steps_per_epoch=4000,max_epoch_len=1000,
                 replay_size=int(1e6),batch_size=128,gamma=0.99,polyak=0.995,lr_a=1e-3,lr_c=1e-3,
                 start_steps=10000,update_after=1000,update_every=50,act_noise=0.1,num_test_episodes=10,
                 save_freq=1,save_path=''):
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
        self.env=env
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        action_bound = [env.action_space.low, env.action_space.high]
        action_bound[0] = action_bound[0].reshape(1, self.act_dim)
        action_bound[1] = action_bound[1].reshape(1, self.act_dim)
        self.action_bound=action_bound
        
        self.epochs=epochs
        self.steps_per_epoch=steps_per_epoch
        self.max_epoch_len=max_epoch_len
        self.replay_size=replay_size
        self.batch_size=batch_size
        self.gamma=gamma
        self.polyak=polyak
        self.start_steps=start_steps
        self.update_after=update_after
        self.update_every=update_every
        self.num_test_episodes=num_test_episodes
        
        self.act_noise=act_noise
        
        self.save_freq=save_freq
        self.save_path=save_path
        
        self.actor=self.build_actor_net(h_layers)
        self.critic=self.build_critic_net(h_layers)
        self.actor_target=self.build_actor_net(h_layers)
        self.critic_target=self.build_critic_net(h_layers)
        self.assign_params(self.actor, self.actor_target)
        self.assign_params(self.critic, self.critic_target)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=lr_a)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=lr_c)
        
    # assign params of p1 to p2
    def assign_params(self,p1,p2):
        for i,j in zip(p1.trainable_weights,p2.trainable_weights):
            j.assign(i)
            
    def update_targets(self):
        for ref,target in zip(self.actor.trainable_weights,self.actor_target.trainable_weights):
            target.assign(self.polyak*target+(1-self.polyak)*ref)
        for ref,target in zip(self.critic.trainable_weights,self.critic_target.trainable_weights):
            target.assign(self.polyak*target+(1-self.polyak)*ref)
        
    def build_actor_net(self,h_layers,activation=tf.nn.relu):
        inputs = Input(shape=[self.state_dim, ])
        for i in range(len(h_layers)):
            if i == 0:
                x = Dense(h_layers[i], activation=activation)(inputs)
            else:
                x = Dense(h_layers[i], activation=activation)(x)
        pi = Dense(self.act_dim, activation=tf.nn.tanh)(x)
        pi = (pi + 1.) * (self.action_bound[1] - self.action_bound[0]) / 2. + self.action_bound[0]
        return tf.keras.Model(inputs=inputs, outputs=pi)
    
    def build_critic_net(self,h_layers,activation=tf.nn.relu):
        state_input=Input(shape=[self.state_dim,])
        action_input=Input(shape=[self.act_dim,])
        x=tf.concat([state_input,action_input],axis=-1)
        for i in range(len(h_layers)):
            x = Dense(h_layers[i], activation=activation)(x)
        y=Dense(1,activation=None)(x)
        return tf.keras.Model(inputs=[state_input,action_input], outputs=y)
    
    @tf.function
    def learn(self,batch):
        states=batch['obs1']
        states_=batch['obs2']
        acts=batch['acts']
        rews=batch['rews'][:,np.newaxis]
        dones=batch['done'][:,np.newaxis]
        
        with tf.GradientTape() as tape:
            acts_=self.actor_target(states_)
            q_=self.critic_target([states_,acts_])
            y=rews+self.gamma*(1-dones)*q_
            q=self.critic([states,acts])
            td_error=tf.losses.mean_squared_error(y,q)
        critic_grads=tape.gradient(td_error,self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_weights))
            
        with tf.GradientTape() as tape:
            acts_pred=self.actor(states)
            q=self.critic([states,acts_pred])
            actor_loss=-tf.math.reduce_mean(q)
        actor_grads=tape.gradient(actor_loss,self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        
    @tf.function
    def sample_action(self,obs,is_explore=True):
        a = self.actor(obs[np.newaxis,:])
        if is_explore:
            a += self.act_noise*tf.random.normal(tf.shape(a))
        # return shape: (1,1)
        a = tf.clip_by_value(a, self.action_bound[0], self.action_bound[1])
        return a[0]
    
    def train(self):
        replay_buffer=ReplayBuffer(self.state_dim, self.act_dim, self.replay_size)
        
        total_steps = self.steps_per_epoch * self.epochs
        
        state = self.env.reset()
        state = state.astype(np.float32)
        ep_len, ep_rew, ep_count = 0,0,0
        all_ep_rew= []
        for t in range(total_steps):
            # randomly sample actions untils start_steps have elapsed
            if t > self.start_steps:
                act = self.sample_action(state)
                act = act.numpy()
            else:
                act = self.env.action_space.sample()
            
            state_, r, d, _ = self.env.step(act)
            state_ = state_.astype(np.float32)
            d = False if ep_len==self.max_epoch_len else d
            ep_len+=1
            ep_rew+=r
            
            # Store transitions
            replay_buffer.store(state,act,r,state_,d)
            
            state = state_
            
            # End of trajectory
            if d or (ep_len==self.max_epoch_len):
                state = self.env.reset()
                state = state.astype(np.float32)
                
                if len(all_ep_rew) < 5:
                    all_ep_rew.append(ep_rew)
                else:
                    all_ep_rew.append(ep_rew)
                    all_ep_rew[-1] = (np.mean(all_ep_rew[-5:]))  # smoothing
                epoch=(t+1)//self.steps_per_epoch
                print("Training | Epoch:{} | Episode:{}  | Steps: {}/{} | Episode Reward: {:.4f}".format(epoch, ep_count, t, total_steps, ep_rew))
                
                ep_len, ep_rew = 0,0
                ep_count += 1
            
            # Update
            if t>self.update_after and t%self.update_every==0:
                for _ in range(self.update_every):
                    batch=replay_buffer.sample_batch(self.batch_size)
                    self.learn(batch)
                    self.update_targets()
            
            # End of epoch
            if (t+1) % self.steps_per_epoch==0:
                epoch=(t+1)//self.steps_per_epoch
                
                # save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.actor.save_weights(self.save_path+'actor_checkpoint'+str(epoch))
                    self.critic.save_weights(self.save_path+'critic_checkpoint'+str(epoch))
        plt.figure()
        plt.plot(all_ep_rew)
        plt.xlabel('episodes')
        plt.ylabel('total reward per episode')
        plt.show()
                
    def test(self,path):
        self.actor.load_weights(path)
    
        while True:
            state = self.env.reset()
            state = state.astype(np.float32)
            
            print("Trying a new epoch...")
    
            while True:
                self.env.render()
                
                # shape of action, state : (1,) and (3,)
                action = self.sample_action(state,is_explore=False) 
                state_, rew, done, _ = self.env.step(action.numpy())
                state_ = state_.astype(np.float32)
                state = state_
    
                if done:
                    break
