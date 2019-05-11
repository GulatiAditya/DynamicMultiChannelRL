"""
Utility functions used in main project
Code Author - Aditya Gulati and Abhinav Gupta
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np 
import tensorflow as tf 
import pandas as pd 
from collections import deque

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow.keras import layers

#From one state,actions and observations make new states
def state_gen(state,action,obs):
    state_out = state[0].tolist()
    state_out.append(action)
    state_out.append(obs)
    state_out = state_out[2:]
    return np.asarray(state_out)


# Fetch states,actions,observations and next state from memory
def get_states(batch): 
    states = []
    for i in batch:
        states.append(i[0])    
    state_arr = np.asarray(states)
    state_arr = state_arr.reshape(32,32)
    return state_arr

def get_actions(batch): 
    actions = []
    for i in batch:
        actions.append(i[1])    
    actions_arr = np.asarray(actions)
    actions_arr = actions_arr.reshape(32)
    return actions_arr

def get_rewards(batch): 
    rewards = []
    for i in batch:
        rewards.append(i[2])
    rewards_arr = np.asarray(rewards)
    rewards_arr = rewards_arr.reshape(1,32)
    return rewards_arr

def get_next_states(batch): 
    next_states = []
    for i in batch:
        next_states.append(i[3])
    next_states_arr = np.asarray(next_states)
    next_states_arr = next_states_arr.reshape(32,32)
    return next_states_arr
