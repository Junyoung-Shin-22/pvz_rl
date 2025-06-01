import gym
from itertools import count
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from pvz import config
import matplotlib.pyplot as plt

HP_NORM = 100
SUN_NORM = 200

class PolicyNetV2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50):
        super(PolicyNetV2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

class HJ_ReinforceAgent():
    def __init__(self, input_size, possible_actions,
                 use_baseline=False, use_entropy=False, lambda_entropy=0.01):
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.possible_actions = possible_actions
        self.policy = PolicyNetV2(input_size, output_size=len(possible_actions))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.n_plants = 4

        self.use_baseline = use_baseline
        self.use_entropy = use_entropy
        self.lambda_entropy = lambda_entropy

    def decide_action(self, observation):
        mask = self._get_mask(observation)
        var_s = torch.from_numpy(observation.astype(np.float32))

        with torch.no_grad():
            log_probs = self.policy(var_s)
            probs = torch.exp(log_probs)

            probs[~mask] = 0
            total = torch.sum(probs)

            if total == 0 or torch.isnan(total):
                # 유효한 행동이 없거나 확률이 NaN 
                valid_actions = np.array(self.possible_actions)[mask]
                if len(valid_actions) == 0:
                    return 0  # 기본 대기 행동 등
                return np.random.choice(valid_actions)

            probs /= total
            return np.random.choice(self.possible_actions, p=probs.numpy())

    def discount_rewards(self,r,gamma):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.shape[0])):
            running_add = running_add * gamma + r[t][0]
            discounted_r[t][0] = running_add
        return discounted_r

    def iterate_minibatches(self, observation, actions, rewards, batchsize, shuffle=False):
        assert len(observation) == len(actions)
        assert len(observation) == len(rewards)

        indices = np.arange(len(observation))
        if shuffle:
            np.random.shuffle(indices)
        #import pdb; pdb.set_trace()
        for start_idx in range(0, len(observation), batchsize):
            if shuffle:
                excerpt = indices[start_idx:min(start_idx + batchsize, len(indices))]
            yield observation[excerpt], actions[excerpt], rewards[excerpt]

    def update(self, observation, actions, rewards):
        rewards = self.discount_rewards(rewards, gamma=0.9)
        self.optimizer.zero_grad()
        loss = 0

        for obs_batch, act_batch, rew_batch in self.iterate_minibatches(
            observation, actions, rewards, batchsize=100, shuffle=True):

            obs_batch = np.atleast_2d(obs_batch)
            act_batch = np.atleast_1d(act_batch).reshape(-1)
            rew_batch = np.atleast_1d(rew_batch).reshape(-1)

            mask_list = [self._get_mask(s) for s in obs_batch]
            mask_batch = torch.tensor(mask_list, dtype=torch.bool)

            if mask_batch.ndim == 1:
                mask_batch = mask_batch.unsqueeze(0)

            valid_indices = mask_batch.any(dim=1)

            # 에러 발생 방지........
            if isinstance(valid_indices, torch.Tensor) and valid_indices.ndim == 0:
                valid_indices = valid_indices.unsqueeze(0)

            if not valid_indices.any():
                continue

            obs_valid = obs_batch[valid_indices.numpy()]
            act_valid = act_batch[valid_indices.numpy()]
            rew_valid = rew_batch[valid_indices.numpy()]
            mask_valid = mask_batch[valid_indices]

            s_var = torch.from_numpy(obs_valid.astype(np.float32))
            a_var = torch.from_numpy(act_valid).view(-1).long()
            r_var = torch.from_numpy(rew_valid.astype(np.float32)).view(-1)

            logits = self.policy.fc2(F.leaky_relu(self.policy.fc1(s_var)))
            masked_logits = logits.clone()
            for i in range(len(masked_logits)):
                masked_logits[i, ~mask_valid[i]] = float('-inf')

            log_probs = F.log_softmax(masked_logits, dim=1)
            probs = torch.exp(log_probs)
            selected_log_probs = log_probs[range(len(a_var)), a_var]
            selected_log_probs = torch.nan_to_num(selected_log_probs, nan=0.0)

            if self.use_baseline:
                baseline = r_var.mean()
                advantage = r_var - baseline
            else:
                advantage = r_var

            reinforce_loss = -selected_log_probs * advantage

            if self.use_entropy:
                entropy = -(probs * log_probs).sum(dim=1)
                total_loss = reinforce_loss - self.lambda_entropy * entropy
            else:
                total_loss = reinforce_loss

            loss += total_loss.mean()

        loss.backward()
        self.optimizer.step()

    def save(self, nn_name):
        torch.save(self.policy, nn_name)

    def load(self, nn_name):
        self.policy = torch.load(nn_name)

    def _get_mask(self, observation):
        empty_cells = np.nonzero((observation[:self._grid_size]==0).reshape(config.N_LANES, config.LANE_LENGTH))
        mask = np.zeros(len(self.possible_actions), dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + config.N_LANES * empty_cells[1]) * self.n_plants

        available_plants = observation[-self.n_plants:]
        for i in range(len(available_plants)):
            if available_plants[i]:
                idx = empty_cells + i + 1
                mask[idx] = True
        return mask

class PlayerV2():
    def __init__(self,render=True, max_frames = 1000, n_iter = 100000):
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self.render = render
        self._grid_size = config.N_LANES * config.LANE_LENGTH

        
    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return config.N_LANES * config.LANE_LENGTH + config.N_LANES + len(self.env.plant_deck) + 1

    def num_actions(self):
        return self.env.action_space.n

    def _transform_observation(self, observation):
        observation = observation.astype(np.float64)
        observation_zombie = self._grid_to_lane(observation[self._grid_size:2*self._grid_size])
        observation = np.concatenate([observation[:self._grid_size], observation_zombie, 
        [observation[2 * self._grid_size]/SUN_NORM], 
        observation[2 * self._grid_size+1:]])
        if self.render:
            print(observation)
        return observation

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1)/HP_NORM

    def play(self,agent, epsilon=0):
        """ Play one episode and collect observations and rewards """

        summary = dict()
        summary['rewards'] = list()
        summary['observations'] = list()
        summary['actions'] = list()
        observation = self._transform_observation(self.env.reset())
        
        t = 0

        while(self.env._scene._chrono<self.max_frames):
            if(self.render):
                self.env.render()
            if np.random.random()<epsilon:
                # print("exploration")
                action = np.random.choice(self.get_actions(), 1)[0]
            else:
                action = agent.decide_action(observation)

            summary['observations'].append(observation)
            summary['actions'].append(action)
            observation, reward, done, info = self.env.step(action)
            observation = self._transform_observation(observation)
            summary['rewards'].append(reward)

            if done:
                break

        summary['observations'] = np.vstack(summary['observations'])
        summary['actions'] = np.vstack(summary['actions'])
        summary['rewards'] = np.vstack(summary['rewards'])
        return summary

    def get_render_info(self):
        return self.env._scene._render_info