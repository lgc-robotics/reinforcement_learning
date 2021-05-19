from ppo import PPO
import gym

if __name__ == '__main__':
    env_id = 'LunarLanderContinuous-v2' 
    env = gym.make(env_id)

    agent = PPO(env, h_layers=[32, 32, 32], seed=0, steps_per_epoch=4000, epochs=150, gamma=0.99, lam=0.95,
               clip_ratio=0.2, lr_a=1e-4, lr_c=1e-3, train_a_iters=80, train_c_iters=80,
               max_ep_len=1000, kl_target=0.01, ent_weight=0.001, save_freq=100, save_path='./checkpoints/')

    # training
    agent.train()

    # test
    path = './checkpoints/actor_checkpoint599'
    agent.test(path)
