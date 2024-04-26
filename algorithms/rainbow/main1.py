import argparse
from datetime import datetime
import numpy as np
import torch
from algorithms.rainbow.agent import Agent
from algorithms.rainbow.env import Env, FrameStackEnv
from algorithms.rainbow.memory import ReplayMemory
from algorithms.rainbow.test import test
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='ai2thor', help='ATARI game or environment')
parser.add_argument('--max-num-steps', type=int, default=int(50e6), metavar='STEPS',
                    help='Number of training steps')
parser.add_argument('--max-episode-length', type=int, default=int(1e3), metavar='LENGTH',
                    help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=1, metavar='T',
                    help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE',
                    help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.8, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--num-atoms', type=int, default=51, metavar='C',
                    help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V',
                    help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V',
                    help='Maximum of value distribution support')
parser.add_argument('--model-path', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=1, metavar='k',
                    help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.1, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n',
                    help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ',
                    help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE',
                    help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η',
                    help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε',
                    help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE',
                    help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate-only', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=1e5, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--log-interval', type=int, default=200, metavar='STEPS',
                    help='Number of training steps between logging status')
parser.add_argument('--render', action='store_true', default=False,
                    help='Display screen (testing only)')
parser.add_argument('--config-file', type=str, default='config_files/rainbow_example.json',
                    help='Config file used for ai2thor environment definition')

if __name__ == '__main__':
    args = parser.parse_args()
    print('-' * 10 + '\n' + 'Options' + '\n' + '-' * 10)
    for k, v in vars(args).items():
        print(' ' * 4 + k + ': ' + str(v))
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = False
    else:
        args.device = torch.device('cpu')

    def log(s):
        print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)

    if args.game == 'ai2thor':
        env = FrameStackEnv(AI2ThorEnv(config_file=args.config_file), args.history_length, args.device)
        args.resolution = env.config['resolution']
        args.img_channels = env.observation_space.shape[0]
    else:
        env = Env(args)
        env.train()
        args.resolution = (84, 84)
        args.img_channels = 1
    action_space = env.action_space

    dqn = Agent(args, env)
    mem = ReplayMemory(args, args.memory_capacity)
    priority_weight_increase = (1 - args.priority_weight) / (args.max_num_steps - args.learn_start)
    val_mem = ReplayMemory(args, args.evaluation_size)
    mem_steps, done = 0, True
    for mem_steps in range(args.evaluation_size):
        if done:
            state, done = env.reset(), False
        next_state, _, done, _ = env.step(env.action_space.sample())
        val_mem.append(state, None, None, done)
        state = next_state

    if args.evaluate_only:
        dqn.eval()
        avg_reward, avg_Q = test(env, mem_steps, args, dqn, val_mem, evaluate_only=True)
        print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
    else:
        dqn.train()
        num_steps, done, episode = 0, True, 0
        episode_rewards = []
        episode_Q_values = []
        while num_steps < args.max_num_steps:
            if done:
                state, done = env.reset(), False
            if num_steps % args.replay_frequency == 0:
                dqn.reset_noise()
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)
            mem.append(state, action, reward, done)
            num_steps += 1

            if num_steps % args.log_interval == 0:
                log('num_steps = ' + str(num_steps) + ' / ' + str(args.max_num_steps))

            if num_steps >= args.learn_start:
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if num_steps % args.replay_frequency == 0:
                    dqn.learn(mem)

                if num_steps % args.evaluation_interval == 0:
                    dqn.eval()
                    avg_reward, avg_Q = test(env, num_steps, args, dqn, val_mem)
                    log('num_steps = ' + str(num_steps) + ' / ' + str(args.max_num_steps) +
                        ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                    dqn.train()

                if num_steps % args.target_update == 0:
                    dqn.update_target_net()

            state = next_state

            # Calculate episode rewards and Q-values
            episode_rewards.append(reward)
            with torch.no_grad():
                q_values = dqn.act(state)
                if isinstance(q_values, int):
                    q_values = torch.tensor([q_values], dtype=torch.float32, device=args.device)
            episode_Q_values.append(q_values.mean().item())

            # Check if the episode is finished
            if done:
                # Calculate average reward for the episode
                avg_episode_reward = sum(episode_rewards) / len(episode_rewards)
                print(f"Episode {episode}: Average Reward: {avg_episode_reward}")
                episode_rewards = []

                # Plot episode rewards and Q-values
                plt.plot(range(episode + 1), episode_rewards, label='Average Episode Reward')
                plt.plot(range(episode + 1), episode_Q_values, label='Average Q-value')
                plt.xlabel('Episode')
                plt.ylabel('Value')
                plt.title('Episode Rewards and Q-values')
                plt.legend()
                plt.show()

                episode += 1

    env.close()
