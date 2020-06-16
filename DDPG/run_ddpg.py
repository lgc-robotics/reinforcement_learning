from ddpg import DDPG
import gym

if __name__ == '__main__':
    env_id = 'Pendulum-v0' #Pendulum-v0, MountainCarContinuous-v0
    env = gym.make(env_id)
    agent=DDPG(env,h_layers=[64,64],seed=0,epochs=10,steps_per_epoch=4000,max_epoch_len=1000,
               replay_size=int(1e5),batch_size=128,gamma=0.99,polyak=0.995,lr_a=1e-3,lr_c=1e-3,
               start_steps=10000,update_after=1000,update_every=50,act_noise=0.1,num_test_episodes=10,
               save_freq=1,save_path='./checkpoints/')
    
    # training
    #agent.train()
    
    #test
    path='./checkpoints/actor_checkpoint10'
    agent.test(path)
