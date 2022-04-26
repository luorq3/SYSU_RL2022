import os
import random
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent


class QNetwork(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(QNetwork, self).__init__()
        ##################
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        ##################

    def forward(self, inputs):
        ##################
        return self.network(inputs)
        ##################


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device):
        ##################
        self.buffer_size = buffer_size

        self.obs_shape = observation_space.shape
        self.action_dim = action_space.n

        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=observation_space.dtype)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False
        self.device = device
        ##################

    def __len__(self):
        ##################
        if self.full:
            return self.buffer_size
        return self.pos
        ##################

    def push(self, *transition):
        ##################
        self.observations[self.pos] = np.array(transition[0]).copy()
        self.actions[self.pos] = np.array(transition[1]).copy()
        self.rewards[self.pos] = np.array(transition[2]).copy()
        self.next_observations[self.pos] = np.array(transition[3]).copy()
        self.dones[self.pos] = np.array(transition[4]).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
        ##################

    def sample(self, batch_size):
        ##################
        upper_bound = self.buffer_size if self.full else self.pos
        batch_idx = np.random.randint(0, upper_bound, size=batch_size)

        data = (
            self.observations[batch_idx, :],
            self.actions[batch_idx, :],
            self.rewards[batch_idx, :],
            self.next_observations[batch_idx, :],
            self.dones[batch_idx, :]
        )

        return [torch.tensor(d).to(self.device) for d in data]
        ##################

    def clean(self):
        ##################
        self.full = False
        self.pos = 0
        ##################


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        self.run_name = f"{args.env_name}_dqn_{int(time.time())}"
        self.log_dir = f'logs/{self.run_name}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(f"{self.log_dir}")

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else 'cpu')

        self.batch_size = args.batch_size
        self.total_timesteps = args.total_timesteps
        self.gamma = args.gamma
        self.learning_freq = args.learning_freq
        self.grad_norm_clip = args.grad_norm_clip
        self.target_update_freq = args.target_update_freq
        self.start_e = args.start_e
        self.end_e = args.end_e
        self.exploration_step = args.exploration_fraction * self.total_timesteps
        self.learning_starts = args.learning_starts

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.q_network = QNetwork(args.hidden_size, env.action_space.n).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=args.lr)
        self.target_network = QNetwork(args.hidden_size, env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = ReplayBuffer(args.buffer_size, self.env.observation_space, self.env.action_space, self.device)

        ##################

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self, global_step):
        """
        Implement your training algorithm here
        """
        ##################
        observations, actions, rewards, next_observations, dones = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            target_max, _ = self.target_network(next_observations).max(dim=1)
            td_target = rewards + self.gamma * target_max * (1 - dones)

        actions = actions.unsqueeze(dim=1)
        q_val = self.q_network(observations).gather(1, actions).squeeze()
        loss = F.mse_loss(td_target, q_val)

        if global_step % 100 == 0:
            self.writer.add_scalar("losses/td_loss", loss, global_step)
            self.writer.add_scalar("losses/q_values", q_val.mean().item(), global_step)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.q_network.parameters()), self.grad_norm_clip)
        self.optimizer.step()

        # update the target network
        if global_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        ##################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def linear_schedule(self, global_step):
        slope = (self.end_e - self.start_e) / self.exploration_step
        return max(slope * global_step + self.start_e, self.end_e)

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################

        episodic_len = 0
        episodic_return = 0

        obs = self.env.reset()
        for global_step in range(self.total_timesteps):
            episodic_len += 1

            epsilon = self.linear_schedule(global_step)
            if random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                logits = self.q_network(torch.tensor(obs).unsqueeze(0).to(self.device))
                action = torch.argmax(logits, dim=1).cpu().numpy()[0]

            next_obs, reward, done, _ = self.env.step(action)
            episodic_return += reward

            if done:
                self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                self.writer.add_scalar("charts/episodic_length", episodic_len, global_step)

                print(f"global_step={global_step}, episodic_return={episodic_return}, "
                      f"episodic_len={episodic_len}")

                episodic_len = 0
                episodic_return = 0

            self.replay_buffer.push(obs, action, reward, next_obs.copy())
            obs = next_obs

            if global_step > self.learning_starts and global_step % self.learning_freq == 0:
                self.train(global_step)

        self.env.close()
        self.writer.close()
        ##################
