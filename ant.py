import gym
import numpy as np
from goal_env import mujoco
# Create the AntMaze environment
env = gym.make("AntMaze-v2")

# Reset the environment to get the initial state
obs = env.reset()

# Run the agent for a fixed number of steps
for _ in range(1000):
    # Sample a random action from the action space
    action = env.action_space.sample()
    print(action)
    # Take a step in the environment with the sampled action
    obs, reward, done, info = env.step(action)
    
    # Render the environment (optional)
    env.render()
    
    # Check if the episode is done
    if done:
        # If the episode is done, reset the environment to start a new episode
        obs = env.reset()

# Close the environment
env.close()

