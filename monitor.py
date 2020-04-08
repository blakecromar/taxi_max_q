from collections import deque
import sys

import math
import numpy as np
import os
from time import sleep

def frame(env, reward, samp_reward, step, episode, show_episodes):
    """ Create frame for displaying to console.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - reward (int): the reward of the current step
    - samp_reward (int): the current total reward for episode
    - step (int): the time step of the episode
    - episode (int): how many episodes have elapesed
    - show_episodes (list): List of episodes that will be displayed
    """
    # Intro Message
    if episode in show_episodes and step == 0:
        os.system('cls')
        pixel = "+"
        inside_width = 30
        title = "Episode: {}".format(episode)
        print(pixel + pixel*inside_width + pixel)
        print(pixel + str.center(title, inside_width) + pixel)
        print(pixel + pixel*inside_width + pixel)
            
        sleep(2.000)
    
    # Individual episode rules
    if episode == show_episodes[-3]:
        statement = ""
        delay = 0.001   
    elif episode == show_episodes[-2]:
        statement = ""
        delay = 0.100
    else:
        statement = ""
        delay = 0.300 
    
    # If episode is terminated early
    if step == 200:
        statement = "FAIL: Juice, you still have lots to learn."
        delay = 1.000 
    
    # If episode is completed successfully
    if reward == 20:
        statement = "SUCCESS: Juice, you did it, man!"
        delay = 1.000
    
    os.system('cls')
    print("Reward for step {}: {}".format(step, reward))
    print("Cumulative reward for episode: {}".format(samp_reward))
    print()
    env.render()
    print()
    print(statement)
    sleep(delay)
    

def interact(env, agent, num_episodes=20000, window=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    
    show_episodes = [1, 500, 20000]
    
    step = 0
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        
        # Initial frame
        if i_episode in show_episodes:
            frame(env, 0, samp_reward, 0, i_episode, show_episodes)

        while True:                
            # agent selects an action
            action = agent.select_action(state)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # agent performs internal updates based on sampled experience
            agent.step(state, action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            
            # We've now completed a step
            step += 1
            
            # Running the frame
            if i_episode in show_episodes:
                frame(env, reward, samp_reward, step, i_episode, show_episodes)
            
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                step = 0 # Reset step count
                break
        
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        
        # monitor progress
        moniter = False
        if moniter == True:
            print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
            sys.stdout.flush()
            # check if task is solved (according to OpenAI Gym)
            if best_avg_reward >= 9.7:
                print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
                break
            if i_episode == num_episodes: print('\n')

    return avg_rewards, best_avg_reward