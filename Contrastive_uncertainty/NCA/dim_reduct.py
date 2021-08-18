"""NCA for linear dimensionality reduction.
"""


import matplotlib.pyplot as plt
import numpy as np
import torch

from Contrastive_uncertainty.NCA.nca import NCA_Module
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_circle(r, num_samples):
  t = np.linspace(0, 2*np.pi, num_samples)
  xc, yc = 0, 0  # circle center coordinates
  x = r*np.cos(t) + 0.2*np.random.randn(num_samples) + xc
  y = r*np.sin(t) + 0.2*np.random.randn(num_samples) + yc
  return x, y


def gen_data(num_samples, num_classes, mean, std):
  """Generates the data.
  """
  num_samples_per = num_samples // num_classes
  X = []
  y = []
  for i, r in enumerate(range(num_classes)):
    # first two dimensions are that of a circle
    x1, x2 = make_circle(r+1.5, num_samples_per)
    # third dimension is Gaussian noise
    x3 = std*np.random.randn(num_samples_per) + mean
    X.append(np.stack([x1, x2, x3]))
    y.append(np.repeat(i, num_samples_per))
  X = np.concatenate(X, axis=1)
  y = np.concatenate(y)
  indices = list(range(X.shape[1]))
  np.random.shuffle(indices)
  X = X[:, indices]
  y = y[indices]
  X = X.T  # make it (N, D)
  return X, y


def plot(Xs, y, labels, save=None):
  fig, axes = plt.subplots(1, len(labels), figsize=(18, 4))
  for ax, X, lab in zip(axes, Xs, labels):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    ax.title.set_text(lab)
  if save is not None:
    filename = "./assets/{}".format(save)
    plt.savefig(filename, format="png", dpi=300, bbox_inches='tight')
  #plt.show()


def main():
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    print("[*] Using cpu.")
    device = torch.device("cpu")

  num_samples = 500
  sigma= 5
  X, y = gen_data(num_samples, 5, 0, 5)
  print("data", X.shape)

  # plot first two dimensions of original data
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
  plt.show()

  # fit NCA
  X = torch.from_numpy(X).float().to(device)
  y = torch.from_numpy(y).long().to(device)
  nca = NCA_Module(dim=2, max_iters=1000, tol=1e-5)
  nca.train(X, y, batch_size=None, weight_decay=20)
  X_nca = nca(X).detach().cpu().numpy()

  # plot PCA vs NCA
  y = y.detach().cpu().numpy()
  X = X.detach().cpu().numpy()

  A = nca.A.detach().cpu().numpy()
  print("\nSolution: \n", A)



main()