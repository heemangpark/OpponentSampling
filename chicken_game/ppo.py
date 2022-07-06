import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from environment import *


def one_hot_padding(state):
    pre_vector = [[], []]
    vector = []
    for i1, i2 in zip(state[0], state[1]):
        pre_vector[0] = [1, 0] if i1 == 'sw' else [0, 1]
        pre_vector[1] = [1, 0] if i2 == 'sw' else [0, 1]
        vector.append(pre_vector[0] + pre_vector[1])
    for _ in range(env().max_t - len(vector)):
        vector.append([0, 0, 0, 0])
    return torch.Tensor(np.array(vector, dtype=np.int32).reshape(-1, 4))


# def one_hot_wornn(state):
#     pre_vector = [[], []]
#     one_hot = []
#     for i1, i2 in zip(state[0], state[1]):
#         pre_vector[0] = [1, 0] if i1 == 'sw' else [0, 1]
#         pre_vector[1] = [1, 0] if i2 == 'sw' else [0, 1]
#         one_hot = one_hot + (pre_vector[0] + pre_vector[1])
#     return torch.Tensor(one_hot)


class buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_done = []

    def push(self, state, action, log_prob, reward, is_done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.is_done.append(is_done)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_done[:]


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.in_size, self.hide_size = 4, 16
        self.wx = nn.Parameter(torch.randn(self.hide_size, self.in_size))
        self.wh = nn.Parameter(torch.randn(self.hide_size, self.hide_size))
        self.b = nn.Parameter(torch.zeros(self.hide_size, 1))
        self.layer1 = nn.Linear(self.hide_size, 64)
        self.layer2 = nn.Linear(64, 2)

    def forward(self, states):
        if len(states.size()) == 3:
            final = torch.zeros(1, self.hide_size)
            for encoded_state_batch in states:
                hidden, hidden_list = torch.zeros(self.hide_size, 1), torch.empty(1, self.hide_size)
                for encoded_state in encoded_state_batch:
                    if list(encoded_state) != [0, 0, 0, 0]:
                        hidden = torch.tanh(
                            torch.matmul(self.wx, encoded_state.unsqueeze(-1)) + torch.matmul(self.wh, hidden) + self.b)
                        hidden_list = torch.cat((hidden_list, torch.transpose(hidden, 0, 1)))
                        hidden_list = hidden_list[-1].unsqueeze(0)
                    else:
                        break
                final = torch.cat((final, hidden_list))
            to_mlp = final[1:]
        else:
            hidden, hidden_list = torch.zeros(self.hide_size, 1), torch.empty(1, self.hide_size)
            for encoded_state in states:
                if list(encoded_state) != [0, 0, 0, 0]:
                    hidden = torch.tanh(
                        torch.matmul(self.wx, encoded_state.unsqueeze(-1)) + torch.matmul(self.wh, hidden) + self.b)
                    hidden_list = torch.cat((hidden_list, torch.transpose(hidden, 0, 1)))
                    hidden_list = hidden_list[-1].unsqueeze(0)
                else:
                    break
            to_mlp = hidden_list

        activation = torch.tanh(self.layer1(to_mlp))
        pi = self.layer2(activation)
        return torch.softmax(pi, -1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.in_size, self.hide_size = 4, 16
        self.wx = nn.Parameter(torch.randn(self.hide_size, self.in_size))
        self.wh = nn.Parameter(torch.randn(self.hide_size, self.hide_size))
        self.b = nn.Parameter(torch.zeros(self.hide_size, 1))
        self.layer1 = nn.Linear(self.hide_size, 64)
        self.layer2 = nn.Linear(64, 1)

    def forward(self, states):
        if len(states.size()) == 3:
            final = torch.zeros(1, self.hide_size)
            for encoded_state_batch in states:
                hidden, hidden_list = torch.zeros(self.hide_size, 1), torch.empty(1, self.hide_size)
                for encoded_state in encoded_state_batch:
                    if list(encoded_state) != [0, 0, 0, 0]:
                        hidden = torch.tanh(
                            torch.matmul(self.wx, encoded_state.unsqueeze(-1)) + torch.matmul(self.wh, hidden) + self.b)
                        hidden_list = torch.cat((hidden_list, torch.transpose(hidden, 0, 1)))
                        hidden_list = hidden_list[-1].unsqueeze(0)
                    else:
                        break
                final = torch.cat((final, hidden_list))
            to_mlp = final[1:]
        else:
            hidden, hidden_list = torch.zeros(self.hide_size, 1), torch.empty(1, self.hide_size)
            for encoded_state in states:
                if list(encoded_state) != [0, 0, 0, 0]:
                    hidden = torch.tanh(
                        torch.matmul(self.wx, encoded_state.unsqueeze(-1)) + torch.matmul(self.wh, hidden) + self.b)
                    hidden_list = torch.cat((hidden_list, torch.transpose(hidden, 0, 1)))
                    hidden_list = hidden_list[-1].unsqueeze(0)
                else:
                    break
            to_mlp = hidden_list

        activation = torch.relu(self.layer1(to_mlp))
        v = self.layer2(activation)
        return v

# class Actor(nn.Module):
#     def __init__(self):
#         super(Actor, self).__init__()
#         self.in_size = 0
#         self.layer1 = nn.Linear(self.in_size, 64)
#         self.layer2 = nn.Linear(64, 2)
#
#     def batch_pi(self, states):
#         pi_list = []
#         for i in range(len(states)):
#             self.in_size = 4 * (i+1)
#             self.layer1 = nn.Linear(self.in_size, 64)
#             activation = torch.relu(self.layer1(states[i]))
#             pi = self.layer2(activation)
#             pi_list.append(pi)
#             output = torch.softmax(pi, -1)
#         return output
#
#     def pi(self, states):
#         self.in_size = len(states)
#         self.layer1 = nn.Linear(self.in_size, 64)
#         activation = torch.relu(self.layer1(states))
#         pi = self.layer2(activation)
#         output = torch.softmax(pi, -1)
#         return output
#
#
# class Critic(nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.in_size = 0
#         self.layer1 = nn.Linear(self.in_size, 64)
#         self.layer2 = nn.Linear(64, 1)
#
#     def batch_v(self, states):
#         v_list = torch.empty(0)
#         for i in range(len(states)):
#             self.in_size = 4 * (i+1)
#             self.layer1 = nn.Linear(self.in_size, 64)
#             activation = torch.relu(self.layer1(states[i]))
#             v = self.layer2(activation)
#             v_list = torch.cat((v_list, v))
#         return v_list
#
#     def v(self, states):
#         self.in_size = len(states)
#         self.layer1 = nn.Linear(self.in_size, 64)
#         activation = torch.relu(self.layer1(states))
#         v = self.layer2(activation)
#         return v


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.actor = Actor()
        self.critic = Critic()

    def act(self, state):
        action_probs = self.actor(one_hot_padding(state))
        action = Categorical(action_probs).sample()
        action_log_prob = Categorical(action_probs).log_prob(action)
        return action.detach(), action_log_prob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        action_log_prob = Categorical(action_probs).log_prob(action)
        entropy = Categorical(action_probs).entropy()
        v = self.critic(state)
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
        self.state_values, self.entropy = 0, 0

    def select_action(self, state):
        with torch.no_grad():
            action, action_log_prob = self.previous_policy.act(state)
        return action.item(), action_log_prob.item()

    def update(self):
        rtg = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_done)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rtg.insert(0, discounted_reward)
        rtg = torch.tensor(rtg, dtype=torch.float32)
        rtg_norm = (rtg - rtg.mean()) / (rtg.std() + 1e-10)

        rollout_states = []
        for i in range(len(self.buffer.states)):
            rollout_states.append(one_hot_padding(self.buffer.states[i]))
        rollout_states = torch.squeeze(torch.stack(rollout_states)).detach()
        rollout_actions = torch.squeeze(torch.Tensor(self.buffer.actions)).detach()
        rollout_log_probs = torch.squeeze(torch.Tensor(self.buffer.log_probs)).detach()

        for _ in range(self.epoch):
            log_probs, state_values, entropy = self.policy.evaluate(rollout_states, rollout_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(log_probs - rollout_log_probs.detach())
            ratios = torch.unsqueeze(ratios, 0)
            advantages = rtg_norm - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
            # self.actor_loss = -torch.min(surr1, surr2).mean()
            # self.critic_loss = 0.5 * self.mse(rtg_norm, state_values)
            self.loss = - torch.min(surr1, surr2) + 0.5 * self.mse(rtg_norm, state_values) - 0.01 * entropy

            self.optimizer_actor.zero_grad(), self.optimizer_critic.zero_grad()
            self.loss.mean().backward()
            self.optimizer_actor.step(), self.optimizer_critic.step()

        # self.episode_actor_loss = self.actor_loss.item()
        # self.episode_critic_loss = self.critic_loss.item()
        self.episode_loss = self.loss.mean().item()
        self.state_values, self.entropy = state_values.mean(), entropy.mean()
        self.previous_policy.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path):
        torch.save(self.previous_policy.state_dict(), path)

    def load(self, path):
        self.previous_policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
