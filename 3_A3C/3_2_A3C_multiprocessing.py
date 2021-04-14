import gym
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical


class CartpolePolicy(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(CartpolePolicy, self).__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.softmax(x, dim=-1)

        return x

    def get_action_logprob(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = torch.unsqueeze(obs, 0)
        output = self.forward(obs)
        categorical = Categorical(output)
        action = categorical.sample()
        logprob = categorical.log_prob(action)

        return action.item(), logprob

    def get_weights(self):
        weights = []
        for param in self.parameters():
            weights.append(param.data.clone())

        return weights

    def get_grad(self):
        grads = []
        for name, param in self.named_parameters():
            grads.append(param.grad)

        return grads

    def set_grad(self, grads):
        for target_param, grad in zip(self.parameters(), grads):
            target_param._grad = grad


class CartpoleCritic(nn.Module):
    def __init__(self, obs_dim):
        super(CartpoleCritic, self).__init__()

        self.obs_dim = obs_dim

        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x

    def get_value(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = torch.unsqueeze(obs, 0)
        output = self.forward(obs)

        return output

    def get_weights(self):
        weights = []
        for param in self.parameters():
            weights.append(param.data.clone())

        return weights

    def get_grad(self):
        grads = []
        for name, param in self.named_parameters():
            grads.append(param.grad)

        return grads

    def set_grad(self, grads):
        for target_param, grad in zip(self.parameters(), grads):
            target_param._grad = grad


"""
https://github.com/MorvanZhou/pytorch-A3C/blob/master/shared_adam.py
"""
class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


def test(pi_model):
    env = gym.make("CartPole-v0")
    while True:
        time.sleep(0.1)
        test_episode_reward = 0
        obs = env.reset()
        while True:
            action, _ = pi_model.get_action_logprob(obs)
            next_obs, rew, done, _ = env.step(action)
            test_episode_reward += rew
            obs = next_obs
            if done:
                break
        if test_episode_reward >= 100.0:
            print("test episode reward :", test_episode_reward)


def train(g_pi, g_v, g_pi_optim, g_v_optim, lock):
    t_max = 200
    gamma = 0.99

    # Make env
    env = gym.make("CartPole-v0")

    # shape
    obs_shape = env.observation_space.shape
    act_shape = tuple([1])  # int to Tuple
    obs_dim = obs_shape[0]
    n_actions = env.action_space.n

    while True:
        pi = CartpolePolicy(obs_dim=obs_dim, n_actions=n_actions)
        v = CartpoleCritic(obs_dim=obs_dim)
        pi.train()
        v.train()

        # Synchronize thread-specific parameters
        for target_param, param in zip(pi.parameters(), g_pi.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(v.parameters(), g_v.parameters()):
            target_param.data.copy_(param.data)

        logprob_list = []
        reward_list = []
        obs_list = []

        # rollout until t_max or done
        obs = env.reset()
        for t in range(t_max):
            action, logprob = pi.get_action_logprob(obs)
            next_obs, rew, done, _ = env.step(action)
            logprob_list.append(logprob)
            reward_list.append(rew)
            obs_list.append(obs)
            obs = next_obs
            if done:
                break
        # R
        if done:
            R = 0
        else:
            R = v.get_value(next_obs).item()

        # Reset gradients
        pi.zero_grad()
        v.zero_grad()

        for i in range(len(reward_list) - 2, -1, -1):  # reverse
            R = reward_list[i] + gamma * R
            # accumulate gradient of pi
            pi_advantage = R - v.get_value(obs_list[i]).item()
            pi_loss = -1 * logprob_list[i] * pi_advantage
            pi_loss.backward()

            v_advantage = R - v.get_value(obs_list[i]).squeeze(1)
            v_loss = v_advantage.pow(2).mean()
            v_loss.backward()

        # Update global model
        with lock:
            g_pi.zero_grad()
            g_v.zero_grad()

            g_pi.set_grad(pi.get_grad())
            g_v.set_grad(v.get_grad())

            g_pi_optim.step()
            g_v_optim.step()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    num_workers = 5

    # Make env
    env = gym.make("CartPole-v0")

    # shape
    obs_shape = env.observation_space.shape
    act_shape = tuple([1])  # int to Tuple
    obs_dim = obs_shape[0]
    n_actions = env.action_space.n

    # Define NN
    g_pi = CartpolePolicy(obs_dim=obs_dim, n_actions=n_actions)
    g_v = CartpoleCritic(obs_dim=obs_dim)
    g_pi.share_memory()
    g_v.share_memory()

    # Optimizer
    g_pi_optim = SharedAdam(g_pi.parameters())
    g_v_optim = SharedAdam(g_v.parameters())

    lock = mp.Lock()

    processes = []
    p = mp.Process(target=test, args=(g_pi,))
    p.start()
    processes.append(p)

    for _ in range(num_workers):
        p = mp.Process(target=train, args=(g_pi, g_v, g_pi_optim, g_v_optim, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
