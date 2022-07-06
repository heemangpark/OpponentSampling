from ppo import *
agent_a, agent_ap = PPO(), PPO()
agent_a.load("trained/fictitious_recent_40_11/a/PPO_train_episode_1.pth")
agent_ap.load("trained/fictitious_recent_40_11/a/PPO_train_episode_100.pth")

for i, j in zip(agent_a.named_parameters(), agent_ap.named_parameters()):
    print(i, j)