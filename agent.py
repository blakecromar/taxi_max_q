import numpy as np
from collections import defaultdict
import random
import math

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA)) 
        self.eps = 1.0
        self.decay = 0.999
        self.gamma = 1.0
        self.alpha = 0.1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.eps:
            # Returns the index of the first maximum value 
            return np.argmax(self.Q[state])
        else:
            # A random action is selected from a list of all moves                   
            return random.choice(np.arange(self.nA))
        
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled 
        tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Updating Q value
        current = self.Q[state][action]
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        target = reward + (self.gamma * Qsa_next) # temperal difference
        new_value = current + (self.alpha * (target - current))
        self.Q[state][action] = new_value
        
        # updating Epsilon if the episode is over
        if done:
            self.eps *= self.decay
        