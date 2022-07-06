import os
import math
import random
from ppo import *
# from environment import *
from environment_11 import *

env = env()
agent_a = PPO()
agent_b = copy.deepcopy(agent_a)
max_episode = 30000

for episode in range(max_episode):
    if episode == 0:
        pass
    else:
        random_index = random.choice(range(math.ceil(0.6 * len(os.listdir("trained/fictitious_recent_40_11/b"))),
                                           len(os.listdir("trained/fictitious_recent_40_11/b")) + 1))
        agent_b.load("trained/fictitious_recent_40_11/b/PPO_train_episode_{}.pth".format(random_index))

    action_a, action_a_log_prob = agent_a.select_action()
    action_b, action_b_log_prob = agent_b.select_action()
    reward_a, reward_b = env.step(action_a, action_b)

    agent_a.buffer.push(action_a, action_a_log_prob, reward_a)
    agent_b.buffer.push(action_b, action_b_log_prob, reward_b)

    agent_a.update()
    agent_a.save("trained/fictitious_recent_40_11/a/PPO_train_episode_" + "{}.pth".format(episode + 1))
    agent_a.save("trained/fictitious_recent_40_11/b/PPO_train_episode_" + "{}.pth".format(episode + 1))

    print("Episode: {}".format(episode + 1), "\n",
          "A choose action '{}'".format(['swerve', 'straight'][action_a]), "\n",
          "B choose action '{}'".format(['swerve', 'straight'][action_b]), "\n",
          "A received reward {}".format(reward_a), "\n", "B received reward {}".format(reward_b), "\n")
