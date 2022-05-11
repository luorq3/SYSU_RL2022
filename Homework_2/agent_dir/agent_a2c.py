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


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        ##################
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        ##################

    def forward(self, inputs):
        ##################
        x = F.relu(self.fc1(inputs))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        ##################
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        ##################

    def forward(self, inputs):
        ##################
        x = F.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x


class AgentA2C(Agent):
    def __init__(self, env, args):
        """
                Initialize every things you need here.
                For example: building your model
                """
        super(AgentA2C, self).__init__(env)
        ##################
        str_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_name = f"{args.seed}_a2c_{str_time}"
        self.log_dir = f'logs/{args.env_name}/{self.run_name}'
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
        self.grad_norm_clip = args.grad_norm_clip
        self.batch_size = args.batch_size

        self.env.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.actor = Actor(np.prod(self.env.observation_space.shape), args.hidden_size,
                           self.env.action_space.n).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.lr)

        self.critic = Critic(np.prod(self.env.observation_space.shape), args.hidden_size).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr)

        self.episodic_lens = []
        self.episodic_returns = []
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
        pass
        ##################

    def make_action(self, obs, test=False):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        logits = self.actor(torch.tensor(obs).unsqueeze(0).to(self.device))[0]
        if test:
            return logits.argmax().item()

        action = Categorical(logits).sample().item()
        prob = logits[action]

        return action, prob
        ##################

    def roll_out(self, s):

        episodic_len = 0
        episodic_return = 0

        o_list, a_list, r_list, mask_list = [], [], [], []
        is_done = False
        v_final = 0
        obs = s
        final_obs = s

        for _ in range(self.batch_size):
            a, _ = self.make_action(obs)
            obs_, r, done, _ = self.env.step(a)
            o_list.append(obs)
            a_list.append(a)
            r_list.append(r)
            mask_list.append(1 - done)

            episodic_len += 1
            episodic_return += r

            final_obs = obs_
            obs = obs_
            if done:
                is_done = True
                self.episodic_lens.append(episodic_len)
                self.episodic_returns.append(episodic_return)
                episodic_len = 0
                episodic_return = 0
                obs = self.env.reset()

        if not is_done:
            v_final = self.critic(torch.tensor(final_obs).unsqueeze(0).to(self.device))[0]

        return o_list, a_list, r_list, mask_list, v_final, obs, is_done

    def compute_target(self, v_final, r_list, mask_list):
        td_target = []
        discount_return = v_final
        for reward, mask in zip(r_list[::-1], mask_list[::-1]):
            discount_return = reward + self.gamma * discount_return * mask
            td_target.append(discount_return)

        return td_target

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        obs = self.env.reset()
        for global_step in range(1, self.total_timesteps + 1):

            o_list, a_list, r_list, mask_list, v_final, obs, is_done = self.roll_out(obs)

            if is_done:
                for episodic_return, episodic_len in zip(self.episodic_returns, self.episodic_lens):
                    self.writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    self.writer.add_scalar("charts/episodic_length", episodic_len, global_step)

                    print(f"global_step={global_step}, episodic_return={episodic_return}, "
                          f"episodic_len={episodic_len}")
                self.episodic_lens.clear()
                self.episodic_returns.clear()

            td_target = self.compute_target(v_final, r_list, mask_list)

            td_target = torch.tensor(td_target).to(self.device)
            o_list = torch.tensor(np.array(o_list)).to(self.device)
            a_list = torch.tensor(np.array(a_list)).unsqueeze(dim=1).to(self.device)
            advantage = td_target - self.critic(o_list).squeeze()

            pi = self.actor(o_list)
            pi_a = pi.gather(1, a_list)
            actor_loss = -(torch.log(pi_a) * advantage).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            critic_loss = F.smooth_l1_loss(self.critic(o_list).squeeze(), td_target)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self.env.close()
        self.writer.close()
        ##################
