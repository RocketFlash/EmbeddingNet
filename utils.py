from sklearn.manifold import TSNE
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import yaml
from keras import optimizers
from augmentations import get_aug
from data_loader import SiameseImageLoader

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


def parse_net_params(filename='configs/road_signs.yml'):
    params = {}
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    
    if cfg['learning_rate']:
        learning_rate = cfg['learning_rate']
    else:
        learning_rate = 0.0004

    if cfg['optimizer'] == 'adam':
        optimizer = optimizers.Adam(lr=learning_rate)
    elif cfg['optimizer'] == 'rms_prop':
        optimizer = optimizers.RMSprop(lr=learning_rate)
    else:
        optimizer = optimizers.SGD(lr=learning_rate)

    if 'augmentations_type' in cfg:
        augmentations = get_aug(cfg['augmentation_type'], cfg['input_shape'])
    else:
        augmentations = None
    

    params = {k:v for k,v in cfg.items() if k not in ['optimizer']}
    params['encodings_path'] = os.path.join(cfg['encodings_path'], 
                                            cfg['project_name'])
    params['plots_path'] = os.path.join(cfg['plots_path'], 
                                        cfg['project_name'])
    params['tensorboard_log_path'] = os.path.join(cfg['tensorboard_log_path'], 
                                                  cfg['project_name'])
    params['weights_save_path'] = os.path.join(cfg['weights_save_path'], 
                                               cfg['project_name'])

    if 'dataset_path' in cfg:
        params['loader'] = SiameseImageLoader(cfg['dataset_path'], 
                                    input_shape=cfg['input_shape'], 
                                    augmentations=augmentations)

    params['optimizer'] = optimizer
    return params