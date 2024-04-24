"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

Contains the train code run by each A3C process on either Atari or AI2ThorEnv.
For initialisation, we set up the environment, seeds, shared model and optimizer.
In the main training loop, we always ensure the weights of the current model are equal to the
shared model. Then the algorithm interacts with the environment args.num_steps at a time,
i.e it sends an action to the env for each state and stores predicted values, rewards, log probs
and entropies to be used for loss calculation and backpropagation.
After args.num_steps has passed, we calculate advantages, value losses and policy losses using
Generalized Advantage Estimation (GAE) with the entropy loss added onto policy loss to encourage
exploration. Once these losses have been calculated, we add them all together, backprop to find all
gradients and then optimise with Adam and we go back to the start of the main training loop.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from algorithms.a3c.envs import create_atari_env
from algorithms.a3c.model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

save_dir = "cups-rl/algorithms/a3c/saved_models"
def save_model_and_metrics(model, optimizer, metrics, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it does not exist

    # List all files in the directory and filter by model files
    existing_files = [f for f in os.listdir(save_dir) if f.startswith('model_') and f.endswith('.pth')]
    if existing_files:
        # Extract model numbers and find the maximum number
        existing_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
        next_model_number = max(existing_numbers) + 1
    else:
        next_model_number = 1  # Start from 1 if no existing model

    # Form the filename for the new model
    model_filename = f'model_{next_model_number}.pth'
    save_path = os.path.join(save_dir, model_filename)

    # Save the model and metrics
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)

    print(f"Training complete. Model and metrics saved to {save_path}")



def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    if args.atari:
        env = create_atari_env(args.atari_env_name)
    else:
        args.config_dict = {'max_episode_length': args.max_episode_length}
        env = AI2ThorEnv(config_dict=args.config_dict)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.frame_dim)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

     # Initialize metrics storage
    metrics = {
        'episode_rewards': [],
        'policy_losses': [],
        'value_losses': [],
    }

    # monitoring
    episode_count = 0
    episode_rewards = []
    policy_losses = []
    value_losses = []

    episode_length = 0
    max_episodes = 1000
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
            if episode_rewards:  # New episode
                episode_count += 1
                episode_total_reward = sum(episode_rewards)
                print(f"Episode {episode_count}. Total Length: {len(episode_rewards)}. Total Reward: {episode_total_reward}.")

                metrics['episode_rewards'].append(episode_total_reward)
                metrics['policy_losses'].append(sum(policy_losses) / len(policy_losses) if policy_losses else 0)
                metrics['value_losses'].append(sum(value_losses) / len(value_losses) if value_losses else 0)


                if episode_count >= max_episodes:  # Condition to end training
                    save_model_and_metrics(model, optimizer, metrics,save_dir )
                    break   
                episode_rewards, policy_losses, value_losses = [], [], []        
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0).float(), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            action_int = action.numpy()[0][0].item()
            state, reward, done, _ = env.step(action_int)

            rewards.append(reward)
            episode_rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            done = done or episode_length >= args.max_episode_length

            with lock:
                counter.value += 1

            if done:
               
                state = env.reset()
                episode_length = 0
                print('Step no: {}'.format(episode_length))

            state = torch.from_numpy(state)

            if done:
                break

        # Backprop and optimisation
        R = torch.zeros(1, 1)
        if not done:  # to change last reward to predicted value to ....
            value, _, _ = model((state.unsqueeze(0).float(), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        # import pdb;pdb.set_trace() # good place to breakpoint to see training cycle
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - \
                          args.entropy_coef * entropies[i]
        
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
