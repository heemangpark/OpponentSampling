from ppo import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if __name__ == "__main__":
    agent_a = PPO()
    action_output = []
    for index in range(30000):
        if index % 100 == 0:
            agent_a.load('trained/fictitious_recent_40/a/PPO_train_episode_' + str(index + 1) + '.pth')
            action_prob = agent_a.policy.actor(torch.Tensor([0, 0, 0, 0]))
            # action_output.append([np.exp(action_log_prob), 1 - np.exp(action_log_prob)])
            action_output.append(action_prob.detach().numpy())
            print(action_output[-1])

    # pca = PCA(n_components=2)
    # latent_2d_pca = pca.fit_transform(action_output)
    action_output = np.array(action_output)
    axes = plt.gca()
    plt.scatter(action_output[:, 1], action_output[:, 0], c=range(300), cmap='viridis')
    plt.ylabel('Swerve')
    plt.xlabel('Straight')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.colorbar()
    plt.savefig("pca_fictitious_40.png")
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    plt.close("all")
