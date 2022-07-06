import pickle
import matplotlib.pyplot as plt
from ppo import *
from sklearn.decomposition import PCA


def pickleloader(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = []
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            return data


chicken = PPO()
inp = chicken.load("trained/PPO_train_episode_10000.pth")
file = open('data.pkl', 'ab')
pickle.dump(inp, file)
file.close()

data_all = pickleloader('data.pkl')
output = []

for policy_id in range(500, 10001, 500):
    i = chicken.load("trained/PPO_train_episode_" + str(policy_id) + ".pth")
    output.append(i)

pca = PCA(n_components=2)
latent_2d_pca = pca.fit_transform(i)

axes = plt.gca()
plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=range(500, 19400, 100), cmap='viridis')
plt.colorbar()
plt.savefig("pca_velocity_.png")
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
plt.close("all")
