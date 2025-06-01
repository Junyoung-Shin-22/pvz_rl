import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pvz import config
from torch.distributions import Categorical

HP_NORM = 100
SUN_NORM = 200

class PolicynetAC3(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=80):
        super(PolicynetAC3, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        return action_prob

# input_size = 55, output_size = 181, for this case
class SJYPolicynetAC3(nn.Module):
    def __init__(self, input_size, output_size):
        super(SJYPolicynetAC3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)

class ValuenetAC3(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=80):
        super(ValuenetAC3, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.leaky_relu(self.affine1(x))
        state_value = self.value_head(x)
        return state_value

# input_size = 55, output_size = 1 (obviously), for this case
class SJYValuenetAC3(nn.Module):
    def __init__(self, input_size):
        super(SJYValuenetAC3, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)

class ACAgent3():
    def __init__(self, input_size, possible_actions, gamma=0.99, device='cuda'):
        self.possible_actions = possible_actions
        output_size = len(possible_actions)
        self.gamma = gamma

        self.device = torch.device(device)
        # self.policy_net = PolicynetAC3(input_size, output_size=len(possible_actions))
        # self.value_net = ValuenetAC3(input_size, output_size=len(possible_actions))
        self.policy_net = SJYPolicynetAC3(input_size, output_size).to(self.device)
        self.value_net = SJYValuenetAC3(input_size).to(self.device)

        self.optimizer_p = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.optimizer_v = optim.Adam(self.value_net.parameters(), lr=1e-4)
        self.saved_actions = []

    def decide_action(self, state):
        state = torch.from_numpy(state).to(dtype=torch.float, device=self.device)
        probs = self.policy_net(state)
        value = self.value_net(state)
        
        # Create a categorical distribution over the list of probabilities of actions
        # Sample an action using the distribution
        m = Categorical(probs)
        action = m.sample()
        
        self.saved_actions.append((m.log_prob(action), value)) # Save to action buffer
        return action.item() # Return the action to take

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        discounted_r[-1] = r[-1]

        for t in range(r.shape[0]-2, -1, -1):
            discounted_r[t] = self.gamma * discounted_r[t+1] + r[t]
        return discounted_r

    def update(self, rewards):
        # Discount rewards through the whole episode
        discounted_rewards = self.discount_rewards(rewards)
        discounted_rewards = torch.from_numpy(discounted_rewards).to(dtype=torch.float, device=self.device)

        saved_actions = self.saved_actions
        policy_losses = [] # List to save actor (policy) loss
        value_losses = [] # List to save critic (value) loss
        self.optimizer_p.zero_grad()
        self.optimizer_v.zero_grad()

        # Store the losses
        for (log_prob, value), R in zip(saved_actions, discounted_rewards):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, R))

        # Compute both losses and backpropagate
        loss_p = torch.stack(policy_losses).sum()
        loss_p.backward(loss_p)
        self.optimizer_p.step()
        
        loss_v = torch.stack(value_losses).sum()
        loss_v.backward(loss_v)
        self.optimizer_v.step()
        
        self.saved_actions = []

    def save(self, nn_name_1, nn_name_2):
        torch.save(self.policy_net, nn_name_1)
        torch.save(self.value_net, nn_name_2)

    def load(self, nn_name_1, nn_name_2):
        self.policy_net = torch.load(nn_name_1)
        self.value_net = torch.load(nn_name_2)


class TrainerAC3():
    def __init__(self, max_frames = 1000):
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self._grid_size = config.N_LANES * config.LANE_LENGTH

    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return config.N_LANES * config.LANE_LENGTH + config.N_LANES + len(self.env.plant_deck) + 1

    def num_actions(self):
        return self.env.action_space.n

    def _transform_observation(self, observation):
        observation_zombie = self._grid_to_lane(observation[self._grid_size:2*self._grid_size])
        observation = np.concatenate([observation[:self._grid_size], observation_zombie,
        [observation[2 * self._grid_size]/SUN_NORM],
        observation[2 * self._grid_size+1:]])

        return observation

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1)/HP_NORM

    def play(self,agent):
        """ Play one episode and collect observations and rewards """

        observations, actions, rewards = [], [], []
        
        observation = self.env.reset()
        while(self.env._scene._chrono < self.max_frames):
            observation = self._transform_observation(observation)
            observations.append(observation)

            action = agent.decide_action(observation)
            actions.append(action)

            observation, reward, done, info = self.env.step(action)
            rewards.append(reward)

            if done: break

        observations = np.vstack(observations)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        return dict(observations=observations, actions=actions, rewards=rewards)

    def get_render_info(self):
        return self.env._scene._render_info