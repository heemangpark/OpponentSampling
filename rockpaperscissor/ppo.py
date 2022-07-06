import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from environment import *

device = torch.device('cpu')


def one_hot_padding(state):
    pre_vector = [[], []]
    vector = []
    for i1, i2 in zip(state[0], state[1]):
        if i1 == 'rock':
            pre_vector[0] = [1, 0, 0]
        elif i1 == 'paper':
            pre_vector[0] = [0, 1, 0]
        elif i1 == 'scissor':
            pre_vector[0] = [0, 0, 1]
        if i2 == 'rock':
            pre_vector[1] = [1, 0, 0]
        elif i2 == 'paper':
            pre_vector[1] = [0, 1, 0]
        elif i2 == 'scissor':
            pre_vector[1] = [0, 0, 1]
        vector.append(pre_vector[0] + pre_vector[1])
    for _ in range(env().maxt - len(vector)):
        vector.append([7, 7, 7, 7, 7, 7])
    return torch.Tensor(np.array(vector, dtype=np.int32).reshape(-1, 6))


class buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_done = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_done[:]


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.in_size, self.hide_size = 6, 12
        self.wx = nn.Parameter(torch.randn(self.hide_size, self.in_size))
        self.wh = nn.Parameter(torch.randn(self.hide_size, self.hide_size))
        self.b = nn.Parameter(torch.zeros(self.hide_size, 1))
        self.layer1 = nn.Linear(self.hide_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 3)

    def forward(self, states):
        hiddens, hidden, count = [], torch.zeros(self.hide_size, 1), 0
        if len(states.size()) == 3:
            outputs = torch.zeros((1, self.hide_size)).to(device)
            for encoded_state_batch in states:
                count = 0
                for encoded_state in encoded_state_batch:
                    if list(encoded_state) != [7, 7, 7, 7, 7, 7]:
                        count += 1
                    hidden = torch.tanh(
                        torch.matmul(self.wx, encoded_state.to('cpu')).unsqueeze(-1) + torch.matmul(self.wh,
                                                                                                    hidden) + self.b)
                    hiddens.append(hidden)
                output = torch.transpose(hiddens[count - 1], 0, 1).to(device)
                outputs = torch.cat((outputs, output))
            outputs = outputs[1:]
        else:
            for encoded_state in states:
                if list(encoded_state) != [7, 7, 7, 7, 7, 7]:
                    count += 1
                hidden = torch.tanh(
                    torch.matmul(self.wx, encoded_state.to('cpu')).unsqueeze(-1) + torch.matmul(self.wh,
                                                                                                hidden) + self.b)
                hiddens.append(hidden)
            outputs = torch.transpose(hiddens[count - 1], 0, 1).to(device)

        activation1 = torch.tanh(self.layer1(outputs))
        activation2 = torch.tanh(self.layer2(activation1))
        pi = self.layer3(activation2)
        return F.softmax(pi, -1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.in_size, self.hide_size = 6, 12
        self.wx = nn.Parameter(torch.randn(self.hide_size, self.in_size))
        self.wh = nn.Parameter(torch.randn(self.hide_size, self.hide_size))
        self.b = nn.Parameter(torch.zeros(self.hide_size, 1))
        self.layer1 = nn.Linear(self.hide_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, states):
        hiddens, hidden, count = [], torch.zeros(self.hide_size, 1), 0
        if len(states.size()) == 3:
            outputs = torch.zeros((1, self.hide_size)).to(device)
            for encoded_state_batch in states:
                count = 0
                for encoded_state in encoded_state_batch:
                    if list(encoded_state) != [7, 7, 7, 7, 7, 7]:
                        count += 1
                    hidden = torch.tanh(
                        torch.matmul(self.wx, encoded_state.to('cpu')).unsqueeze(-1) + torch.matmul(self.wh,
                                                                                                    hidden) + self.b)
                    hiddens.append(hidden)
                output = torch.transpose(hiddens[count - 1], 0, 1).to(device)
                outputs = torch.cat((outputs, output))
            outputs = outputs[1:]
        else:
            for encoded_state in states:
                if list(encoded_state) != [7, 7, 7, 7, 7, 7]:
                    count += 1
                hidden = torch.tanh(
                    torch.matmul(self.wx, encoded_state.to('cpu')).unsqueeze(-1) + torch.matmul(self.wh,
                                                                                                hidden) + self.b)
                hiddens.append(hidden)
            outputs = torch.transpose(hiddens[count - 1], 0, 1).to(device)

        activation1 = torch.tanh(self.layer1(outputs))
        activation2 = torch.tanh(self.layer2(activation1))
        v = self.layer3(activation2)
        return v


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)

    def act(self, state):
        action_probs = self.actor(state)
        action = Categorical(action_probs).sample()
        action_logprob = Categorical(action_probs).log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        action_logprobs = Categorical(action_probs).log_prob(action)
        entropy = Categorical(action_probs).entropy()
        v = self.critic(state)
        return action_logprobs, v, entropy


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.gamma = 0.99
        self.clip = 0.1
        self.epoch = 5
        self.buffer = buffer()
        self.policy = ActorCritic()
        self.optimizer_actor = optim.Adam(params=self.policy.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(params=self.policy.critic.parameters(), lr=0.001)
        self.previous_policy = ActorCritic()
        self.previous_policy.load_state_dict(self.policy.state_dict())
        self.mse = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            action, action_logprob = self.previous_policy.act(one_hot_padding(state))
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_done)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            # rewards: [r1 + 0.99*r2 + 0.99^2*r3 + ... / r2 + 0.99*r3 + 0.99^2*r4 + ... / ...]
            # rewards refers to a list := [Q(s1,a1), Q(s2,a2), Q(s3,a3), ... , Q(sT,aT)]

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        rollout_states = []
        for i in range(len(self.buffer.states)):
            rollout_states.append(one_hot_padding(self.buffer.states[i]))
        rollout_states = torch.squeeze(torch.stack(rollout_states)).detach().to(device)
        rollout_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        rollout_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        for _ in range(self.epoch):
            logprobs, v, entropy = self.policy.evaluate(rollout_states, rollout_actions)
            v = torch.squeeze(v)
            ratios = torch.exp(logprobs - rollout_logprobs.detach())
            advantages = rewards - v.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
            loss = - torch.min(surr1, surr2) + 0.5 * self.mse(v, rewards) - 0.1 * entropy

            self.optimizer_actor.zero_grad(), self.optimizer_critic.zero_grad()
            loss.mean().backward()
            self.optimizer_actor.step(), self.optimizer_critic.step()

        self.previous_policy.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path):
        torch.save(self.previous_policy.state_dict(), path)

    def load(self, path):
        self.previous_policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
