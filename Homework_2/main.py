import argparse
from wrappers import make_env
import gym
from argument import dqn_arguments, pg_arguments, ddpg_arguments


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')
    parser.add_argument('--train_ddpg', default=False, type=bool, help='whether train ddpg')

    parser = dqn_arguments(parser)
    # parser = ddpg_arguments(parser)
    # parser = pg_arguments(parser)
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.run()

    if args.train_dqn:
        env_name = args.env_name
        env = make_env(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.run()

    if args.train_ddpg:
        env_name = args.env_name
        env = make_env(env_name)
        from agent_dir.agent_ddpg import AgentDDPG
        agent = AgentDDPG(env, args)
        agent.run()


if __name__ == '__main__':
    args = parse()
    run(args)
    # env = gym.make(args.env_name)
