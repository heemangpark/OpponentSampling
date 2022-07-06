import wandb
from ppo import *
from environment import *

chicken, agent = env(), PPO()
buffer = agent.buffer
save_term, max_episode = 100, 10000
wandb_plot = False
mixed_heuristic = False

if wandb_plot:
    exp_name = "heuristic\'{}\'".format("mix")
    wandb.init(project="chicken_game", name=exp_name)

for episode in range(max_episode):
    state = chicken.reset()  # s0
    action_a, action_log_prob = agent.select_action(state)  # a1, a1_prob
    count, max_reward, episode_reward, action_b = 0, 0, 0, None
    heuristic_type = random.choice([3, 4, 5]) if mixed_heuristic else 3

    while True:
        if heuristic_type == 1:
            action_b = 0
            max_reward = 96
        elif heuristic_type == 2:
            action_b = 1
            max_reward = 0
        elif heuristic_type == 3:
            action_b = 0 if count % 2 == 0 else 1
            count += 1
            max_reward = 48
        elif heuristic_type == 4:  # b1
            action_b = 0 if count % 3 in [0, 1] else 1
            count += 1
            max_reward = 64
        elif heuristic_type == 5:
            action_b = 0 if count % 4 in [0, 1, 2] else 1
            count += 1
            max_reward = 72
        else:
            assert heuristic_type in [1, 2, 3, 4, 5], "Undefined Heuristic Type"

        # s1, r1, terminal of s1 / s2, r2, terminal of s2 ...
        state, reward, done = chicken.step(action_a, action_b)

        # a2, a2_prob / a3, a3_prob ...
        action_a, action_log_prob = agent.select_action(state)

        # s1, a2, r1(s1,a1), terminal of s1 / s2, a3, r2(s2,a2), terminal of s2
        buffer.push(state, action_a, action_log_prob, reward, done)

        # append reward
        episode_reward += reward

        if chicken.is_done():
            print("Episode: {}\nCurrent Heuristic Type: {}\nCurrent Episode Reward: {} / {}\n"
                  .format(episode + 1, heuristic_type, episode_reward, max_reward))
            agent.update()

            if wandb_plot:
                wandb.log(
                    {"episode": episode + 1, "episode reward": episode_reward, "loss": agent.episode_loss,
                     "v": agent.state_values, "entropy": agent.entropy})
            break

    if (episode + 1) % save_term == 0:
        if mixed_heuristic:
            agent.save("trained/trained_h345/PPO_train_episode_" + "{}.pth".format(episode + 1))
            print("Just Saved Model {} / {}\n".format(int((episode + 1) / save_term), int(max_episode / save_term)))
        else:
            agent.save("trained/trained_h{}/PPO_train_episode_".format(heuristic_type) + "{}.pth".format(episode + 1))
            print("Just Saved Model {} / {}\n".format(int((episode + 1) / save_term), int(max_episode / save_term)))
