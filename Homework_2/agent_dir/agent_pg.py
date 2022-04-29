import os
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent


class PGNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PGNetwork, self).__init__()
        ##################
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        ##################

    def forward(self, inputs):
        ##################
        x = F.relu(self.fc1(inputs))
        x = F.softmax(self.fc2(x), dim=1)
        return x
        ##################


class AgentPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentPG, self).__init__(env)
        ##################
        str_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_name = f"{args.env_name}_dqn_{str_time}"
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

        self.total_timesteps = args.total_timesteps
        self.gamma = args.gamma

        self.env.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.pg_network = PGNetwork(np.prod(self.env.observation_space.shape), args.hidden_size, self.env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.pg_network.parameters(), lr=args.lr)

        self.replay_buffer = []

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

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        discount_return = 0
        self.optimizer.zero_grad()
        for reward, prob in self.replay_buffer[::-1]:
            discount_return = reward + self.gamma * discount_return
            loss = -torch.log(prob) * discount_return
            loss.backward()
        self.optimizer.step()
        self.replay_buffer = []
        ##################

    def make_action(self, obs, test=False):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        logits = self.pg_network(torch.tensor(obs).unsqueeze(0).to(self.device))[0]
        if test:
            return logits.argmax().item()

        action = Categorical(logits).sample().item()
        prob = logits[action].item()

        return action, prob
        ##################

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        episodic_len = 0
        episodic_return = 0

        obs = self.env.reset()
        for global_step in range(1, self.total_timesteps + 1):
            episodic_len += 1

            action, prob = self.make_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            episodic_return += reward

            self.replay_buffer.append((reward, prob))

            if done:
                self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                self.writer.add_scalar("charts/episodic_length", episodic_len, global_step)

                print(f"global_step={global_step}, episodic_return={episodic_return}, "
                      f"episodic_len={episodic_len}")

                self.train()

                episodic_len = 0
                episodic_return = 0

                obs = self.env.reset()
            else:
                obs = next_obs

        self.env.close()
        self.writer.close()
        ##################
