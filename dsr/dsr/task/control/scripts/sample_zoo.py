import sys, os, gym
import numpy as np
import dsr.task.control.utils as U

def sample_data(env_name, algorithm, n_episodes, sample_size):
    #global variables
    global REGRESSION_SEED_SHIFT 
    global N_EPISODES
    global SAMPLE_SIZE 
    REGRESSION_SEED_SHIFT = int(2e6)
    N_EPISODES = n_episodes
    SAMPLE_SIZE = sample_size
    #Make gym environment
    env=gym.make(env_name)
    if "Bullet" in env_name:
       env = U.TimeFeatureWrapper(env)
    #Load model
    U.load_default_model(env_name)
    #RUN episodes
    for j in range(N_EPISODES):
        env.seed(j + REGRESSION_SEED_SHIFT)
        obs=env.reset()
        done = False
        obs_list = []
        action_list = []
        while not done:
            obs_list.append(obs)
            action, _states = U.model.predict(obs)
            obs, rewards, done, info = env.step(action)
            action_list.append(action)
        # Save all action and obs as npz file
        np.savez(env_name+"_action.npz", np.asarray(action_list))
        np.savez(env_name+"_obs.npz", np.asarray(obs_list))
        if len(obs_list)-1 >SAMPLE_SIZE : 
            index = np.random.randint(0, len(obs_list)-1, SAMPLE_SIZE)
            for i in index:
                line=""
                for s in obs_list[i]:
                    line =  line + str(s) +","
                for a in action_list[i]:
                    line =  line + str(a)
                print(j, line)
                f.write(line+"\n")
    f.close()


sample_data("LunarLanderContinuous-v2", "sac", 1000, 100000)
sample_data("InvertedPendulumSwingupBulletEnv-v0", "sac", 1000, 100000)
sample_data("ReacherBulletEnv-v0", "td3", 1000, 100000)
sample_data("HopperBulletEnv-v0", "td3", 1000, 100000) 
