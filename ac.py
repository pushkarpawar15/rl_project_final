import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from goal_env import mujoco

# Define the actor network
class Actor(nn.Module):
    def __init__(self, observation_space, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(observation_space['observation'].shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

# Define the critic network
class Critic(nn.Module):
    def __init__(self, observation_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(observation_space['observation'].shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Create the AntMaze environment
env = gym.make("AntMaze-v1")
observation_space = env.observation_space

# Get the number of actions from the action space
num_actions = env.action_space.shape[0]
print(num_actions)
# Instantiate actor and critic networks
actor = Actor(observation_space, num_actions)
critic = Critic(observation_space)

# Define the optimizer
optimizer_actor = optim.Adam(actor.parameters(), lr=0.001)
optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)

# Main loop
for i in range(10):
    obs = env.reset()
    done = False
    print(i)
    while not done:
        obs_tensor = torch.FloatTensor(obs['observation']).unsqueeze(0)
        
        # Sample action from the actor network
        action_probs = actor(obs_tensor)
        action = np.random.choice(num_actions, p=action_probs.detach().numpy()[0])
        
        # Take a step in the environment
        next_obs, reward, done, _ = env.step(action)
        next_obs_tensor = torch.FloatTensor(next_obs['observation']).unsqueeze(0)
        print(iter,":",reward)
        
        # Compute critic values
        value = critic(obs_tensor)
        next_value = critic(next_obs_tensor)
        
        # Compute TD error
        td_error = reward + 0.99 * next_value - value
        
        # Compute actor loss
        actor_loss = -torch.log(action_probs[0, action]) * td_error
        
        # Compute critic loss
        critic_loss = td_error ** 2
        
        # Zero the gradients before backpropagation
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        
        # Backpropagation for actor loss
        actor_loss.backward(retain_graph=True)
        
        # Update actor
        optimizer_actor.step()
        
        # Zero the gradients for critic
        optimizer_critic.zero_grad()
        
        # Backpropagation for critic loss
        critic_loss.backward(retain_graph=True)
        
        # Update critic
        optimizer_critic.step()
        
        obs = next_obs


# Main loop for testing
for i in range(10):  # You can adjust the number of testing episodes as needed
    obs = env.reset()
    done = False
    while not done:
        # Render the environment during testing
        env.render()
        
        obs_tensor = torch.FloatTensor(obs['observation']).unsqueeze(0)
        
        # Sample action from the actor network without exploration
        with torch.no_grad():
            action_probs = actor(obs_tensor)
            action = torch.argmax(action_probs).item()
        
        # Take a step in the environment
        next_obs, reward, done, _ = env.step(action)
        
        obs = next_obs

# Close the environment after testing
env.close()

# Close the environment
env.close()
