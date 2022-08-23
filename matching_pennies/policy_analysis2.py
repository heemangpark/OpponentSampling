from ppo import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    agent_a = PPO()
    action_output = []
    for index in range(30000):
        if index % 1 == 0:
            agent_a.load('trained/fictitious_recent_100/a/PPO_train_episode_' + str(index + 1) + '.pth')
            action_prob = agent_a.policy.actor(torch.Tensor([0, 0, 0, 0]))
            action_output.append(action_prob.detach().numpy())

    action_output = np.array(action_output)
    axes = plt.gca()
    plt.plot(action_output[:, 0])
    plt.savefig("pca_fictitious_100_interval_6.png")
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    plt.close("all")
