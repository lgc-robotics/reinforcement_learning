from td3 import TD3
import gym

if __name__ == '__main__':
    env_id = 'Pendulum-v0' #Pendulum-v0, MountainCarContinuous-v0
    env = gym.make(env_id)
    agent=TD3(env,h_layers=[64,64],seed=0,steps_per_epoch=4000,epochs=10,max_ep_len=1000,
               batch_size=100,start_steps=10000,update_after=1000,update_every=50,
               replay_size=int(1e5),gamma=0.99,polyak=0.995,lr_a=1e-3,lr_c=1e-3,
               act_noise=0.1,target_noise=0.2,noise_clip=0.5,policy_delay=2,
               save_freq=1,save_path='./checkpoints/')
    
    # training
    agent.train()
    
    #test
    #path='./checkpoints/actor_checkpoint10'
    #agent.test(path)
