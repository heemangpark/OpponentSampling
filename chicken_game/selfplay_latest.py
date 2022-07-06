import wandb
from ppo import *
from environment import *

chicken = env()
agent_a = PPO()
agent_ap = copy.deepcopy(agent_a)
episode_reward_a_list, episode_reward_ap_list = [], []
max_episode = 100000

wandb_plot = True
if wandb_plot:
    exp_name = "self play latest_1"
    wandb.init(project="chicken_game", name=exp_name)

for episode in range(max_episode):
    state = chicken.reset()  # s0
    if episode == 0:
        pass
    else:
        agent_ap.load("trained_sp/latest_1/ap/PPO_train_episode_{}.pth".format(episode))
    action_a, action_a_log_prob = agent_a.select_action(state)  # a1, a1_prob
    action_ap, action_ap_log_prob = agent_ap.select_action(state)  # ap1, ap1_prob
    episode_reward_a, episode_reward_ap = 0, 0

    while True:
        # s1, r1(s1,a1), terminal of s1 / s2, r2(a2, b2), terminal of s2 ...
        state_a, state_ap, reward_a, reward_ap, done = chicken.step_sp(action_a, action_ap)

        # a2, a2_prob & ap2, ap2_prob / a3, a3_prob & ap3, ap3_prob ...
        action_a, action_a_log_prob = agent_a.select_action(state_a)
        action_ap, action_ap_log_prob = agent_ap.select_action(state_ap)

        # s1, a2, r1, terminal of s1 / s2, a3, r2, terminal of s2
        agent_a.buffer.push(state_a, action_a, action_a_log_prob, reward_a, done)
        agent_ap.buffer.push(state_ap, action_ap, action_ap_log_prob, reward_ap, done)

        # append reward
        episode_reward_a += reward_a
        episode_reward_ap += reward_ap

        if chicken.is_done():
            episode_reward_a_list.append(episode_reward_a), episode_reward_ap_list.append(episode_reward_ap)
            print("Episode: {}\nCurrent Episode Reward of A: {}\nCurrent Episode Reward of A`: {}"
                  .format(episode + 1, episode_reward_a, episode_reward_ap))
            print("Average Reward of A So Far: {:.3f}\nAverage Reward of A` So Far: {:.3f}\n"
                  .format(np.mean(episode_reward_a_list), np.mean(episode_reward_ap_list)))
            agent_a.update()
            agent_a.save("trained_sp/latest_1/a/PPO_train_episode_" + "{}.pth".format(episode + 1))
            agent_a.save("trained_sp/latest_1/ap/PPO_train_episode_" + "{}.pth".format(episode + 1))

            if wandb_plot:
                wandb.log(
                    {"average reward of A": np.mean(episode_reward_a_list),
                     "reward of A": episode_reward_a,
                     "average reward of A`": np.mean(episode_reward_ap_list),
                     "reward of A`": episode_reward_ap,
                     "loss of A": agent_a.episode_loss})
            break
