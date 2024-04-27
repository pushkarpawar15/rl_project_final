import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from goal_env import mujoco



# PointMaze

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('PointMaze-v0')  # Assuming PointMaze is the environment name
env.reset()
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 128)  # Assuming the state space of PointMaze is 2-dimensional

        # actor's layer
        self.action_mean = nn.Linear(128, env.action_space.shape[0])  # Output dimension is action space dimension

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: chooses action to take from state s_t
        # by returning mean of the normal distribution
        action_mean = self.action_mean(x)
        action_std = torch.tensor([1e-2])  # Fixed standard deviation for simplicity

        # critic: evaluates being in the state s_t
        state_value = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. the normal distribution representing the action
        # 2. the value from state s_t
        return Normal(action_mean, action_std), state_value


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


# def select_action(state):
#     print(state)

#     state = torch.from_numpy(state).float()
#     action_dist, state_value = model(state)

#     # sample an action from the normal distribution
#     action = action_dist.sample()

#     # save to action buffer
#     model.saved_actions.append(SavedAction(action_dist.log_prob(action), state_value))

#     # the action to take
#     return action.detach().numpy()

def select_action(state):
    observation = state['observation']
    state_values = torch.tensor([observation], dtype=torch.float32)
    action_dist, _ = model(state_values)

    # sample an action from the normal distribution
    action = action_dist.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(action_dist.log_prob(action), None))

    # the action to take
    return action.detach().numpy()

def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 0  # Initialize running reward for PointMaze

    # run infinitely many episodes
    for i_episode in count(1):
        # env.render()
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # Check if we have achieved the goal
        if done:
            print("Goal achieved in episode {} with reward {}".format(i_episode, ep_reward))
            break


if __name__ == '__main__':
    main()
