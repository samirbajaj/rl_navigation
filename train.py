from collections import deque
import argparse
import os
import matplotlib
# pip install PyQt5
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from dqn_agent import Agent


def create_env(file_name="Banana_Linux/Banana.x86_64"):
    env = UnityEnvironment(file_name=file_name)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    #print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    #print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    #print('States look like:', state)
    state_size = len(state)
    #print('States have length:', state_size)

    return env, brain_name, state_size, action_size


def train_dqn(agent, env, weights_file='checkpoint.pth', n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), weights_file)
            break
    return scores


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Navigation - Udacity Reinforcement Learning Nanodegree.')
    parser.add_argument("--train", help="train the Q network", action="store_true")
    parser.add_argument("--weights", help="path to file containing the neural network parameters",
                        default="checkpoint.pth")
    args = parser.parse_args()
    argsdict = vars(args)
    env, brain_name, state_size, action_size = create_env()
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    if args.train:
        if os.path.exists(argsdict['weights']):
            raise Exception(f"Cannot overwrite parameters file {argsdict['weights']}")
        print(f"Commencing training loop...parameters will be saved in {argsdict['weights']}...")
        scores = train_dqn(agent, env)
        plot_scores(scores)
    else:
        if not (os.path.isfile(argsdict['weights']) and os.access(argsdict['weights'], os.R_OK)):
            raise Exception(f"Cannot access parameters from {argsdict['weights']}")
        print(f"Running pre-trained agent using parameters from {argsdict['weights']}...")
        agent.qnetwork_local.load_state_dict(torch.load(argsdict['weights']))

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        print("Score: {}".format(score))
    env.close()
