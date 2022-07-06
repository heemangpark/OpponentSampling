import wandb
from ppo import *
from environment import *

env = env()
agent = PPO()
heuristic_type = 'paper'
max_episode = 10000
save_term = 100
wandb_plot = False

if wandb_plot:
    exp_name = "heuristic\'{}\'".format(heuristic_type)
    wandb.init(project="rockpaperscissor", name=exp_name)

for episode in range(max_episode):
    state = env.reset()
    episode_reward = 0
    b_heuristic = 0

    while True:
        if heuristic_type == 'rock':
            action_b = 0  # heuristic rock (maximum reward per episode: 12)
        elif heuristic_type == 'paper':
            action_b = 1  # heuristic paper (maximum reward per episode: 12)
        elif heuristic_type == 'scissor':
            action_b = 2  # heuristic scissor (maximum reward per episode: 12)
        else:
            assert b_heuristic in ['rock', 'paper', 'scissor'], "Undefined Heuristic Type"

        action_a = agent.select_action(state)
        state, reward, done = env.step(action_a, action_b)
        agent.buffer.rewards.append(reward)
        agent.buffer.is_done.append(done)
        episode_reward += reward

        if env.is_done():
            print("Episode: {}\nCurrent Episode Reward: {}\n".format(episode + 1, episode_reward))
            if wandb_plot:
                wandb.log({"episode": episode + 1, "episode reward": episode_reward})
            break

    agent.update()

    if (episode + 1) % save_term == 0:
        agent.save("trained/PPO_train_episode_" + "{}.pth".format(episode + 1))
        print("Just Saved Model {} / {}\n".format(int((episode + 1) / save_term), int(max_episode / save_term)))
