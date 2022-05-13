import os
import random
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device):
        ##################
        self.buffer_size = buffer_size

        self.obs_shape = observation_space.shape
        self.act_shape = action_space.shape

        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size,) + self.act_shape, dtype=action_space.dtype)
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
            self.actions[batch_idx],
            self.rewards[batch_idx],
            self.next_observations[batch_idx, :],
            self.dones[batch_idx]
        )

        return [torch.tensor(d).to(self.device) for d in data]
        ##################

    def clean(self):
        ##################
        self.full = False
        self.pos = 0
        ##################


class QNetwork(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size):
        super(QNetwork, self).__init__()
        ##################
        self.fc1 = nn.Linear(np.prod(obs_shape) + np.prod(action_shape), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        ##################

    def forward(self, inputs, a):
        ##################
        x = torch.cat([inputs, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        ##################


class Actor(nn.Module):
    def __init__(self, observation_shape, action_shape, hidden_size):
        super(Actor, self).__init__()
        ##################
        self.fc1 = nn.Linear(np.array(observation_shape).prod(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, np.array(action_shape).prod())
        ##################

    def forward(self, inputs):
        ##################
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
        ##################


class AgentDDPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDDPG, self).__init__(env)
        ##################

        assert isinstance(env.action_space, gym.spaces.Box), "Only continuous action space is supported."

        str_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_name = f"{args.env_name}_ddpg_{str_time}"
        self.log_dir = f'logs/{self.run_name}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(f"{self.log_dir}")
        self.writer.add_text(
            "hyper-parameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else 'cpu')

        self.env.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Args
        self.tau = args.tau
        self.gamma = args.gamma
        self.total_timesteps = args.total_timesteps
        self.learning_starts = args.learning_starts
        self.exploration_noise = args.exploration_noise
        self.policy_frequency = args.policy_frequency

        self.max_action = float(env.action_space.high[0])
        self.actor = Actor(env.observation_space.shape, env.action_space.shape, args.hidden_size).to(self.device)
        self.target_actor = Actor(env.observation_space.shape, env.action_space.shape, args.hidden_size).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qv = QNetwork(env.observation_space.shape, env.action_space.shape, args.hidden_size).to(self.device)
        self.target_qv = QNetwork(env.observation_space.shape, env.action_space.shape, args.hidden_size).to(self.device)
        self.target_qv.load_state_dict(self.qv.state_dict())
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.learning_rate)
        self.qv_optimizer = optim.Adam(list(self.qv.parameters()), lr=args.learning_rate)

        self.replay_buffer = ReplayBuffer(args.buffer_size, self.env.observation_space, self.env.action_space, device=self.device)
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
            next_state_action = self.target_actor(next_observations).clamp(min=self.env.action_space.low, max=self.env.action_space.high)
            qf_next_target = self.target_qv(next_observations, next_state_action)
            next_q_value = rewards + (1 - dones) * self.gamma * qf_next_target.view(-1)

        qf_a_values = self.qv(observations, actions).view(-1)
        qf_loss = F.mse_loss(qf_a_values, next_q_value)

        self.qv_optimizer.zero_grad()
        qf_loss.backward()
        nn.utils.clip_grad_norm_(list(self.qv.parameters()), self.max_grad_norm)
        self.qv_optimizer.step()

        if global_step % self.policy_frequency == 0:
            actor_loss = -self.qv(observations, self.actor(observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(list(self.qv.parameters()), self.max_grad_norm)
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qv.parameters(), self.target_qv.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if global_step % 100 == 0:
            self.writer.add_scalar("losses/qv_loss", qf_loss.item(), global_step)
            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            self.writer.add_scalar("losses/qf1_values", qf_a_values.mean().item(), global_step)
        ##################

    def make_action(self, global_step, obs, test=False):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################

        if test:
            return self.actor(torch.Tensor(obs).unsqueeze(dim=0).to(self.device)).tolist()[0]

        if global_step >= self.learning_starts:
            action = self.env.sample()
        else:
            action = self.actor(torch.Tensor(obs).unsqueeze(dim=0).to(self.device)).tolist()[0]
            noise = np.random.normal(0, self.max_action * self.exploration_noise, self.env.action_space.shape)
            action = (action + noise).clip(min=self.env.action_space.low, max=self.env.action_space.high)

        return action
        ##################

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################

        episodic_len = 0
        episodic_return = 0

        obs = self.env.reset()
        for global_step in range(1, self.total_timesteps+1):
            episodic_len += 1

            action = self.make_action(global_step, obs)

            next_obs, reward, done, _ = self.env.step(action)
            episodic_return += reward
            self.replay_buffer.push(obs.copy(), action, reward, next_obs.copy(), done)

            if done:
                self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                self.writer.add_scalar("charts/episodic_length", episodic_len, global_step)

                print(f"global_step={global_step}, episodic_return={episodic_return}, "
                      f"episodic_len={episodic_len}")

                episodic_len = 0
                episodic_return = 0

                obs = self.env.reset()
            else:
                obs = next_obs

            if global_step >= self.learning_starts:
                self.train(global_step)

        self.env.close()
        self.writer.close()
        ##################
