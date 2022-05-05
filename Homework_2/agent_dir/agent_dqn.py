import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent


class QNetwork(nn.Module):
    def __init__(self, hidden_size, output_size, dueling_dqn=True):
        super(QNetwork, self).__init__()
        ##################
        self.dueling_dqn = dueling_dqn
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_size),
            nn.ReLU()
        )
        self.q = nn.Linear(hidden_size, output_size)
        self.v = nn.Linear(hidden_size, 1)
        ##################

    def forward(self, inputs):
        ##################
        logits = self.model(inputs)

        if self.dueling_dqn:
            q, v = self.q(logits), self.v(logits)
            return q - q.mean(dim=1, keepdim=True) + v

        return self.q(logits)
        ##################


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device):
        ##################
        self.buffer_size = buffer_size

        self.obs_shape = observation_space.shape

        self.observations = np.zeros((self.buffer_size,) + self.obs_shape, dtype=observation_space.dtype)
        self.actions = np.zeros(self.buffer_size, dtype=action_space.dtype)
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


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        str_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_name = f"{args.seed}_dqn_{str_time}"
        self.log_dir = f'logs/{args.env_name}/{self.run_name}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(f"{self.log_dir}")
        self.writer.add_text(
            "hyper-parameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # Algo config: double DQN / Dueling DQN
        self.double_dqn = args.double_dqn
        self.dueling_dqn = args.dueling_dqn

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else 'cpu')

        self.batch_size = args.batch_size
        self.total_timesteps = args.total_timesteps
        self.gamma = args.gamma
        self.learning_freq = args.learning_freq
        self.grad_norm_clip = args.grad_norm_clip
        self.start_e = args.start_e
        self.end_e = args.end_e
        self.exploration_step = args.exploration_fraction * self.total_timesteps
        self.learning_starts = args.learning_starts

        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)
        self.env.observation_space.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        self.q_network = QNetwork(args.hidden_size, env.action_space.n, self.dueling_dqn).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)

        # If double_dqn is True
        if self.double_dqn:
            self.target_update_freq = args.target_update_freq
            self.target_network = QNetwork(args.hidden_size, env.action_space.n, self.dueling_dqn).to(self.device)
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
            if self.double_dqn:
                target_max, _ = self.target_network(next_observations).max(dim=1)
            else:
                target_max, _ = self.q_network(next_observations).max(dim=1)
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
        if self.double_dqn and global_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        ##################

    def make_action(self, obs, global_step, test=False):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        if test:
            epsilon = self.end_e
        else:
            slope = (self.end_e - self.start_e) / self.exploration_step
            epsilon = max(slope * global_step + self.start_e, self.end_e)

        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            logits = self.q_network(torch.tensor(obs).unsqueeze(0).to(self.device))
            action = torch.argmax(logits, dim=1).cpu().numpy()[0]

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

            action = self.make_action(obs, global_step)

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

            if global_step >= self.learning_starts and global_step % self.learning_freq == 0:
                self.train(global_step)

        self.env.close()
        self.writer.close()
        ##################
