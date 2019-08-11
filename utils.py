from sklearn.manifold import TSNE
import pickle
import numpy as np
from matplotlib import pyplot as plt


def load_encodings(path_to_encodings):
    
    with open(path_to_encodings, 'rb') as f:
        encodings = pickle.load(f)
    return encodings


def make_tsne(project_name, show=True):
    encodings = load_encodings(
        'encodings/{}encodings.pkl'.format(project_name))
    labels = list(set(encodings['labels']))
    tsne = TSNE()
    tsne_train = tsne.fit_transform(encodings['encodings'])
    fig, ax = plt.subplots(figsize=(16, 16))
    for i, l in enumerate(labels):
        ax.scatter(tsne_train[np.array(encodings['labels']) == l, 0],
                   tsne_train[np.array(encodings['labels']) == l, 1], label=l)
    ax.legend()
    if show:
        fig.show()

    fig.savefig("plots/{}{}.png".format(project_name, 'tsne.png'))
