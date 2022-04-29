def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="BreakoutNoFrameskip-v4", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--buffer_size", default=int(20000), type=int)
    parser.add_argument("--lr", default=0.00025, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=0.5, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--total_timesteps", default=int(4000000), type=int)
    parser.add_argument("--learning_freq", default=1, type=int)
    parser.add_argument("--target_update_freq", default=1000, type=int)

    # Self_defined arguments
    # For epsilon-greedy liner decay
    parser.add_argument("--start_e", default=1, type=float)
    parser.add_argument("--end_e", default=0.01, type=float)
    parser.add_argument("--exploration_fraction", default=0.4, type=float)
    # Off-policy learning begin step
    parser.add_argument("--learning_starts", default=10000, type=int)

    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--total_timesteps", default=int(30000), type=int)

    return parser


def ddpg_arguments(parser):
    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument('--env_name', default="LunarLanderContinuous-v2", help='environment name')
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", default=int(20000), type=int)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--exploration_noise", type=float, default=0.1)
    parser.add_argument("--learning_starts", type=int, default=25e3)
    parser.add_argument("--policy_frequency", type=int, default=2)
    parser.add_argument("--noise_clip", type=float, default=0.5)

    return parser
