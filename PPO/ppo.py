"""
reference:
https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/ppo.py
"""
import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from rl_toolbox.ppo.buffer import PPOBuffer
from rl_toolbox.ppo.distributions import DiagonalGaussian

class PPO(object):
    def __init__(self, env, h_layers=[64, 64], seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, lam=0.97,
                 clip_ratio=0.2, lr_a=3e-4, lr_c=1e-3, train_a_iters=80, train_c_iters=80, max_ep_len=1000, kl_target=0.01, ent_weight=0.001, save_freq=100, save_path=''):

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        action_bound = [env.action_space.low, env.action_space.high]
        action_bound[0] = action_bound[0].reshape(1, self.action_dim)
        action_bound[1] = action_bound[1].reshape(1, self.action_dim)
        self.action_bound = action_bound

        self.steps_per_epoch = steps_per_epoch
        self.max_ep_len = max_ep_len
        self.train_a_iters = train_a_iters
        self.train_c_iters = train_c_iters
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.clip_ratio = clip_ratio
        self.kl_target = kl_target
        self.ent_weight = ent_weight
        self.save_freq = save_freq
        self.save_path = save_path

        self.actor = self.build_actor_net(h_layers)
        self.critic = self.build_critic_net(h_layers)

        # important
        lr_schedule_a = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_a,decay_steps=self.train_a_iters * self.epochs/5., decay_rate=0.96,staircase=True)
        lr_schedule_c = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_c, decay_steps=self.train_c_iters * self.epochs/5., decay_rate=0.96,staircase=True)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_a)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_c)

    def build_actor_net(self, h_layers, activation=tf.nn.relu6):
        inputs = Input(shape=[self.state_dim, ])
        initializer = tf.keras.initializers.Orthogonal()
        for i in range(len(h_layers)):
            if i == 0:
                x = Dense(h_layers[i], activation=activation, kernel_initializer=initializer)(inputs)
            else:
                x = Dense(h_layers[i], activation=activation, kernel_initializer=initializer)(x)
        # mean
        mean = Dense(self.action_dim, activation=None, kernel_initializer=initializer)(x)
        #mean = Dense(self.action_dim, activation=tf.nn.softsign, kernel_initializer=initializer)(x)
        #mean = (mean + 1.) * (self.action_bound[1] - self.action_bound[0]) / 2. + self.action_bound[0]

        # std
        log_std = Dense(self.action_dim, kernel_initializer=initializer)(x)
        std = tf.math.exp(log_std)
        return tf.keras.Model(inputs=inputs, outputs=[mean, std, log_std])

    def build_critic_net(self, h_layers, activation=tf.nn.relu6):
        inputs = Input(shape=[self.state_dim, ])
        initializer = tf.keras.initializers.Orthogonal()
        for i in range(len(h_layers)):
            if i == 0:
                x = Dense(h_layers[i], activation=activation, kernel_initializer=initializer)(inputs)
            else:
                x = Dense(h_layers[i], activation=activation, kernel_initializer=initializer)(x)
        outputs = Dense(1, activation=None, kernel_initializer=initializer)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    @tf.function
    def sample_action(self, observation):
        # shape (1,act_dim)
        mean, std, log_std = self.actor(observation[np.newaxis, :])
        pi = DiagonalGaussian(mean, std, log_std)
        #action = tf.clip_by_value(pi.sample(), self.action_bound[0], self.action_bound[1])
        action = pi.sample()

        # shape (1,1)
        value = self.critic(observation[np.newaxis, :])
        return action[0], value[0, 0]

    # @tf.function
    def update_actor(self, states, actions, advantages, old_pi):
        with tf.GradientTape() as tape:
            mean, std, log_std = self.actor(states)
            pi = DiagonalGaussian(mean, std, log_std)

            log_pi = pi.log_likelihood(actions)
            log_old_pi = old_pi.log_likelihood(actions)
            ratio = tf.exp(log_pi - log_old_pi)
            surr = tf.math.minimum(ratio * advantages,
                                   tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages)
            loss = -tf.math.reduce_mean(surr)

            approx_ent = tf.math.reduce_mean(-log_pi)
            loss -= approx_ent*self.ent_weight # maximize the entropy to encourage exploration

            approx_kl = tf.math.reduce_mean(log_old_pi - log_pi)

        grad = tape.gradient(loss, self.actor.trainable_weights)
        # very important to clip gradient
        grad, grad_norm = tf.clip_by_global_norm(grad, 0.5)
        self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_weights))
        return approx_kl

    # @tf.function
    def update_critic(self, states, returns):
        with tf.GradientTape() as tape:
            advantage = returns - self.critic(states)
            loss = tf.math.reduce_mean(0.5 * tf.square(advantage))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        # very important to clip gradient
        grad, grad_norm = tf.clip_by_global_norm(grad, 0.5)
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.trainable_weights))

    # it should be commented to use 'kl'.
    # @tf.function
    def update(self, states, actions, returns, advantages):
        mean, std, log_std = self.actor(states)
        old_pi = DiagonalGaussian(mean, std, log_std)

        for i in range(self.train_a_iters):
            kl = self.update_actor(states, actions, advantages, old_pi)
            if kl > tf.constant(1.5 * self.kl_target):
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
        for i in range(self.train_c_iters):
            self.update_critic(states, returns)

    def train(self, seed=116):
        self.env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        buffer = PPOBuffer(self.state_dim, self.action_dim, size=self.steps_per_epoch, gamma=self.gamma, lam=self.lam)

        all_episode_reward = []
        episode_count = 0
        for epoch in range(self.epochs):
            episode_reward = 0
            ep_len = 0
            state = self.env.reset()
            state = state.astype(np.float32)
            for ii in range(self.steps_per_epoch):
                # if RENDER:
                #    self.env.render()
                ep_len += 1

                # shape of action, state : (1,) and (3,)
                action, value = self.sample_action(state)
                action = action.numpy()
                value = value.numpy()
                state_, rew, done, _ = self.env.step(action)
                state_ = state_.astype(np.float32)

                buffer.store(state, action, rew, value)
                state = state_

                episode_reward += rew
                if done or ii == self.steps_per_epoch - 1 or ep_len == self.max_ep_len:

                    if done:
                        last_value = 0
                    else:
                        last_value = self.critic(state_[np.newaxis, :])
                        last_value = last_value[0, 0]
                    buffer.finish_path(last_value)

                    episode_count += 1
                    print("Training | episode:{}  | epoch: {} | Episode Reward: {:.4f} | Episode Length: {}".format(episode_count,
                                                                                                  epoch,
                                                                                                  episode_reward,
                                                                                                  ep_len))
                    if (episode_count + 1) % self.save_freq == 0:
                        self.actor.save_weights(self.save_path + 'actor_checkpoint' + str(episode_count))
                        self.critic.save_weights(self.save_path + 'critic_checkpoint' + str(episode_count))

                    if len(all_episode_reward) < 5:  # record running episode reward
                        all_episode_reward.append(episode_reward)
                    else:
                        all_episode_reward.append(episode_reward)
                        all_episode_reward[-1] = (np.mean(all_episode_reward[-5:]))  # smoothing

                    state = self.env.reset()
                    state = state.astype(np.float32)
                    episode_reward = 0
                    ep_len = 0

            state_buf, act_buf, adv_buf, ret_buf = buffer.get()
            state_tensor = tf.convert_to_tensor(np.vstack(state_buf), dtype=tf.float32)
            act_tensor = tf.convert_to_tensor(np.vstack(act_buf), dtype=tf.float32)
            adv_tensor = tf.convert_to_tensor(np.vstack(adv_buf), dtype=tf.float32)
            ret_tensor = tf.convert_to_tensor(np.vstack(ret_buf), dtype=tf.float32)

            adv_tensor = tf.squeeze(adv_tensor, axis=1)

            self.update(state_tensor, act_tensor, ret_tensor, adv_tensor)

        plt.figure()
        plt.plot(all_episode_reward)
        plt.xlabel('episodes')
        plt.ylabel('total reward per episode')
        plt.show()

    def test(self, path):

        self.actor.load_weights(path)
        # self.critic.load_weights(path)

        while True:
            state = self.env.reset()
            state = state.astype(np.float32)

            print("Trying a new epoch...")

            while True:
                self.env.render()

                # shape of action, state : (1,) and (3,)
                action, value = self.sample_action(state)
                state_, rew, done, _ = self.env.step(action.numpy())
                state_ = state_.astype(np.float32)
                state = state_

                if done:
                    break
