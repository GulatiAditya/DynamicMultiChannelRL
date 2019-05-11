"""
main Code for Project
Code Author - Aditya Gulati and Abhinav Gupta
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np 
import tensorflow as tf 
import pandas as pd 
from collections import deque
import utils
import qnetwork
import matplotlib.pyplot as plt
tf.keras.backend.clear_session() 
from tensorflow.keras import layers



#To use Read Trace data
data_in = pd.read_csv("./dataset/real_data_trace.csv")
data_in = data_in.drop("index",axis=1)

#Uncomment to use Perfectly Correlated Scenario
# data_in = pd.read_csv("./dataset/perfectly_correlated.csv")


np.random.seed(40)


# print(len(data_in))
# exit()


TIME_SLOTS = 100000
NUM_CHANNELS = 16               # Number of Channels
memory_size = 1000              # Experience Memory Size
batch_size = 32                 # Batch size for loss calculations (M)
eps = 0.1                       # Exploration Probability
action_size = 16                # Action set size
state_size = 32                 # State Size (a_t-1, o_t-1 ,......, a_t-M,o_t-M)
learning_rate = 1e-2            # Learning rate
gamma = 0.9                     # Discount Factor
hidden_size = 50                # Hidden Size (Put 200 for perfectly correlated)
pretrain_length = 16            # Pretrain Set to be known
n_episodes = 10                 # Number of episodes (equivalent to epochs)


tf.reset_default_graph()

env_model = qnetwork.channel_env(NUM_CHANNELS)      # Intialize Evironment, Network and Batch Memory

q_network = qnetwork.QNetwork(learning_rate=learning_rate,state_size=state_size,action_size=NUM_CHANNELS,hidden_size=hidden_size,name="ChannelQ_Network")

exp_memory = qnetwork.ExpMemory(in_size=memory_size)


history_input = deque(maxlen=state_size)            #Input as states


# Initialise the state of 16 actions and observations with random initialisation

for i in range(pretrain_length):
	action = np.random.choice(action_size)
	obs = data_in["channel"+str(action)][i]
	history_input.append(action)
	history_input.append(obs)


saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

prob_explore = 0.1                  # Exploration Probability

loss_0 = []
avg_loss = []
reward_normalised = []

show_interval = 50                  # To see loss trend iteration-wise (Put this 1 to see full trend)


for episode in range(n_episodes):
    total_rewards = 0
    loss_init = 0
    print("-------------Episode "+str(episode)+"-----------")
    for time in range(len(data_in)-pretrain_length):
        prob_sample = np.random.rand()
        state_in  = np.array(history_input)                 #Start State
        state_in = state_in.reshape([1,-1])
        
        if(prob_sample <= prob_explore ):                   #Exploration
            action = np.random.choice(action_size)
        else:                                               #Exploitation
            action = -1
            
            feed_dict = {q_network.input_in : state_in}
            q_out = sess.run(q_network.out_layer,feed_dict = feed_dict)         # Get qvalue vector from state above
            action = np.argmax(q_out)                                       #Action as argmax qvalue

        obs = data_in["channel"+str(action)][time+pretrain_length]          # Observe
        next_state = utils.state_gen(state_in,action,obs)                   # Go to next state
        reward = obs                                                       # Reward
        total_rewards += reward                                             # Total Reward
        exp_memory.add((state_in,action,reward,next_state))               # Add in exp memory

        state_in = next_state                                              
        history_input = next_state

        if (time>state_size or episode!=0):                              # If sufficient minibatch is available
            batch = exp_memory.sample(batch_size)                       # Sample without replacement  
            states = utils.get_states(batch)                            # Get state,action,reward and next state from memory
            actions = utils.get_actions(batch)
            rewards = utils.get_rewards(batch)
            next_state = utils.get_next_states(batch)

            feed_dict = {q_network.input_in : next_state}
            actuals_Q = sess.run(q_network.out_layer,feed_dict=feed_dict)                # Get the Q values for next state


            actuals = rewards + gamma * np.max(actuals_Q,axis=1)                          # Make it actuals with discount factor       
            actuals = actuals.reshape(batch_size)
            
            # Feed in here to get loss and optimise it
            loss, _  = sess.run([q_network.Q_loss,q_network.opt],feed_dict={q_network.input_in:states,q_network.semiGTq:actuals,q_network.action:actions})


            loss_init += loss 
            
            # We show first episode trend (as it is most drastic)
            if(episode==0):
                loss_0.append(loss)
            
            # Display
            if(time%show_interval == 0):
                print("Loss  at (t="+ str(time) + ") = " + str(loss))

        #Plot Display of Loss in episode 0
        if(time==len(data_in)-pretrain_length-1 and episode==0):
            plt.plot(loss_0)
            plt.xlabel("Iteration")
            plt.ylabel("Q Loss")
            plt.title('Iteration vs Loss (Episode 0)')
            plt.show()

    #Average loss
    print("Average Loss: ")
    print(loss_init/(len(data_in)))
    #Average reward observed in full iterations
    print("Total Reward: ")
    print(total_rewards/len(data_in))
    avg_loss.append(loss_init/(len(data_in)))
    reward_normalised.append(total_rewards/len(data_in))


# See reward and loss trend episode wise

plt.plot(reward_normalised)
plt.xlabel("Episode")
plt.ylabel("Reward Normalised")
plt.title("Episode vs Reward Normalised")
plt.show()

plt.plot(avg_loss)
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.title("Episode vs Average Loss")
plt.show()




exit()


for time in range(len(data_in)-pretrain_length):
    prob_sample = np.random.rand()
    state_in  = np.array(history_input)
    state_in = state_in.reshape([1,-1])
    
    if(prob_sample <= prob_explore ):
        action = np.random.choice(action_size)
    else:
        action = -1
        
        feed_dict = {q_network.input_in : state_in}
        q_out = sess.run(q_network.out_layer,feed_dict = feed_dict)
        action = np.argmax(q_out)

    obs = data_in["channel"+str(action)][time+pretrain_length]
    next_state = utils.state_gen(state_in,action,obs)
    reward = obs
    total_rewards += reward
    exp_memory.add((state_in,action,reward,next_state))

    state_in = next_state
    history_input = next_state

    if time>state_size:
        batch = exp_memory.sample(batch_size)
        states = utils.get_states(batch)
        actions = utils.get_actions(batch)
        rewards = utils.get_rewards(batch)
        next_state = utils.get_next_states(batch)

        feed_dict = {q_network.input_in : next_state}
        actuals_Q = sess.run(q_network.out_layer,feed_dict=feed_dict)


        actuals = rewards + gamma * np.max(actuals_Q,axis=1)
        actuals = actuals.reshape(batch_size)
        loss, _  = sess.run([q_network.Q_loss,q_network.opt],feed_dict={q_network.input_in:states,q_network.semiGTq:actuals,q_network.action:actions})

        print("T = ",time)
        print("Loss = ",loss)

    print(total_rewards)