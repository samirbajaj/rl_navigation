[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, I trained an agent to navigate and collect bananas in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

The solution was developed in Python 3.6.9 on a Linux machine running Ubuntu 18.04. Instructions for setting up the required Python modules are available [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). In addition, `PyQt5` needs to be installed for use by `matplotlib`. 

Finally, [the Unity environment for Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip) should be downloaded and decompressed in the working directory. 

### Training the Agent

The `train.py` script offers a help message:
```commandline
$ python train.py -h
usage: train.py [-h] [--train] [--weights WEIGHTS]

Navigation - Udacity Reinforcement Learning Nanodegree.

optional arguments:
  -h, --help         show this help message and exit
  --train            train the Q network
  --weights WEIGHTS  path to file containing the neural network parameters
``` 
The script can be run in two modes, depending on whether the `--train` argument has been specified.

To train the agent, simply invoke the script with the `--train` argument. Doing so will start the training loop that runs until the desired minimum average score is recorded, or 2,000 training episodes have been completed. If an average score of +13 over 100 consecutive episodes is obtained, training stops and model parameters are written to disk. The default filename for the model parameters is `checkpoint.pth`.

A file with the saved model weights of the successful agent, `model.pt`, is available. It can be used to create and run a pre-trained agent as follows:
```commandline
$ python train.py --weights model.pt
Found path: /home/samir/projects/rl/deep-reinforcement-learning/p1_solution/Banana_Linux/Banana.x86_64
Mono path[0] = '/home/samir/projects/rl/deep-reinforcement-learning/p1_solution/Banana_Linux/Banana_Data/Managed'
Mono config path = '/home/samir/projects/rl/deep-reinforcement-learning/p1_solution/Banana_Linux/Banana_Data/MonoBleedingEdge/etc'
Preloaded 'ScreenSelector.so'
Preloaded 'libgrpc_csharp_ext.x64.so'
Unable to preload the following plugins:
	ScreenSelector.so
	libgrpc_csharp_ext.x86.so
Logging to /home/samir/.config/unity3d/Unity Technologies/Unity Environment/Player.log
INFO:unityagents:
'Academy' started successfully!
Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
		
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
Running pre-trained agent using parameters from model.pt...
Score: 17.0
```

