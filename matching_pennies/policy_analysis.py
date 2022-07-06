from ppo import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    agent_a = PPO()
    action_output = []
    for index in range(10000):
        if index % 10 == 0:
            agent_a.load('trained/fictitious_recent_20/a/PPO_train_episode_' + str(index + 1) + '.pth')
            action_prob = agent_a.policy.actor(torch.Tensor([0, 0, 0, 0]))
            action_output.append(action_prob.detach().numpy())
            print(action_output[-1])

    action_output = np.array(action_output)
    axes = plt.gca()
    plt.scatter(action_output[:, 1], action_output[:, 0], c=range(1000), cmap='viridis')
    plt.ylabel('Head')
    plt.xlabel('Tail')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.colorbar()
    plt.savefig("pca_fictitious_20.png")
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    plt.close("all")
