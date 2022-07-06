import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from environment import *


class buffer:
    def __init__(self):
        self.actions = []
        self.log_probs = []
        self.rewards = []

    def push(self, action, log_prob, reward):
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def clear(self):
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 2)

    def forward(self, start):
        activation = torch.relu(self.layer1(start))
        pi = self.layer2(activation)
        return torch.softmax(pi, -1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 1)

    def forward(self, start):
        activation = torch.relu(self.layer1(start))
        v = self.layer2(activation)
        return v


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.actor = Actor()
        self.critic = Critic()

    def act(self):
        action_probs = self.actor(torch.Tensor([0, 0, 0, 0]))
        action = Categorical(action_probs).sample()
        action_log_prob = Categorical(action_probs).log_prob(action)
        return action.detach(), action_log_prob.detach()

    def evaluate(self, action):
        action_probs = self.actor(torch.Tensor([0, 0, 0, 0]))
        action_log_prob = Categorical(action_probs).log_prob(action)
        v = self.critic(torch.Tensor([0, 0, 0, 0]))
        entropy = Categorical(action_probs).entropy()
        return action_log_prob, v, entropy


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.gamma = 0.99
        self.clip = 0.1
        self.epoch = 5
        self.buffer = buffer()
        self.policy = Policy()
        self.optimizer_actor = optim.Adam(params=self.policy.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(params=self.policy.critic.parameters(), lr=0.001)
        self.previous_policy = Policy()
        self.previous_policy.load_state_dict(self.policy.state_dict())
        self.mse = nn.MSELoss()
        self.actor_loss, self.critic_loss, self.loss = 0, 0, 0
        self.episode_actor_loss, self.episode_critic_loss, self.episode_loss = 0, 0, 0

    def select_action(self):
        with torch.no_grad():
            action, action_log_prob = self.previous_policy.act()
        return action.item(), action_log_prob.item()

    def update(self):
        rollout_action = torch.Tensor(self.buffer.actions).detach()
        rollout_log_prob = torch.Tensor(self.buffer.log_probs).detach()
        rollout_reward = torch.Tensor(self.buffer.rewards).detach()

        for _ in range(self.epoch):
            log_prob, state_value, entropy = self.policy.evaluate(rollout_action)
            state_value = torch.squeeze(state_value)

            ratios = torch.exp(log_prob - rollout_log_prob.detach())
            ratios = torch.unsqueeze(ratios, 0)
            advantages = rollout_reward.detach() - state_value.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
            self.loss = -torch.min(surr1, surr2) + 0.5 * self.mse(rollout_reward.detach(), state_value) - 0.01 * entropy

            self.optimizer_actor.zero_grad(), self.optimizer_critic.zero_grad()
            self.loss.mean().backward()
            self.optimizer_actor.step(), self.optimizer_critic.step()

        self.episode_loss = self.loss.mean().item()
        self.previous_policy.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path):
        torch.save(self.previous_policy.state_dict(), path)

    def load(self, path):
        self.previous_policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
