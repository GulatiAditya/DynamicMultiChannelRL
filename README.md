# DynamicMultiChannelRL
Contains Implementation of Paper " S Wang, H Liu, P H Gomes, and B Krishnamachari ; Deep Reinforcement Learning for Dynamic Multichannel Access in Wireless Networks"


The above repo is final submission of my project for the course " EE 5611 : Machine Learning Applications for Wireless Communications"

Dependencies
Tensorflow-2.0.alpha-0 , numpy, matplotlib, pandas

Some results which are shown in the repository above are (Loss trend in 1st episode , Average loss trend across episodes and reward trend across episodes).

File Descriptions
main.py  - the main source for project
qnetwork.py  - contains environment, network and experience memory classes.
utils.py - utility funcitons for states,action,rewards and next states
dataset - contains realtrace and perfectly correlated data csvs.
graphical results - stored some results as said above.
Slides - Slides used for ppt.

Running Instructions
(Activate your tensorflow environment)
python main.py

