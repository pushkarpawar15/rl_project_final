import gym
import d3rlpy # Import required to register environments

from goal_env import mujoco
# Create the AntMaze environment
env = gym.make("AntMaze-v2")

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
# dataset = d4rl.qlearning_dataset(env)
dataset, env = d3rlpy.datasets.get_d4rl('antmaze')
