from ppo import *
agent_a, agent_b = PPO(), PPO()
agent_a.load("trained/fictitious_recent_20/a/PPO_train_episode_1000.pth")
agent_b.load("trained/fictitious_recent_20/a/PPO_train_episode_3000.pth")

for i, j in zip(agent_a.named_parameters(), agent_b.named_parameters()):
    print(i, j)