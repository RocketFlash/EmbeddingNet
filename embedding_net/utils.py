from sklearn.manifold import TSNE
import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
import yaml
from tensorflow.keras import optimizers
from .augmentations import get_aug
from .data_loader import EmbeddingNetImageLoader
from .datagenerators import TripletsDataGenerator, SiameseDataGenerator, SimpleTripletsDataGenerator


def load_encodings(path_to_encodings):

    with open(path_to_encodings, 'rb') as f:
        encodings = pickle.load(f)
    return encodings


def plot_tsne(encodings_path, save_plot_dir, show=True):
    encodings = load_encodings(encodings_path)
    labels = list(set(encodings['labels']))
    tsne = TSNE()
    tsne_train = tsne.fit_transform(encodings['encodings'])
    fig, ax = plt.subplots(figsize=(16, 16))
    for i, l in enumerate(labels):
        xs = tsne_train[np.array(encodings['labels']) == l, 0]
        ys = tsne_train[np.array(encodings['labels']) == l, 1]
        ax.scatter(xs, ys, label=l)
        for x, y in zip(xs, ys):
            plt.annotate(l,
                         (x, y),
                         size=8,
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

    ax.legend(bbox_to_anchor=(1.05, 1), fontsize='small', ncol=2)
    if show:
        fig.show()

    fig.savefig("{}{}.png".format(save_plot_dir, 'tsne.png'))


def plot_tsne_interactive(encodings_path):
    import plotly.graph_objects as go
    encodings = load_encodings(encodings_path)
    labels = list(set(encodings['labels']))
    tsne = TSNE()
    tsne_train = tsne.fit_transform(encodings['encodings'])
    fig = go.Figure()
    for i, l in enumerate(labels):
        xs = tsne_train[np.array(encodings['labels']) == l, 0]
        ys = tsne_train[np.array(encodings['labels']) == l, 1]
        color = 'rgba({},{},{},{})'.format(int(255*np.random.rand()),
                                           int(255*np.random.rand()),
                                           int(255*np.random.rand()), 0.8)
        fig.add_trace(go.Scatter(x=xs,
                                 y=ys,
                                 mode='markers',
                                 marker=dict(color=color,
                                             size=10),
                                 text=l,
                                 name=l))
    fig.update_layout(
        title=go.layout.Title(text="t-SNE plot",
                              xref="paper",
                              x=0),
        autosize=False,
        width=1000,
        height=1000
    )

    fig.show()


def plot_grapths(history, save_path):
    for k, v in history.history.items():
        t = list(range(len(v)))
        fig, ax = plt.subplots()
        ax.plot(t, v)

        ax.set(xlabel='epoch', ylabel='{}'.format(k),
               title='{}'.format(k))
        ax.grid()

        fig.savefig("{}{}.png".format(save_path, k))

def plot_batch_simple(data, targets, class_names):
        num_imgs = data[0].shape[0]
        img_h = data[0].shape[1]
        img_w = data[0].shape[2]
        full_img = np.zeros((img_h,num_imgs*img_w,3), dtype=np.uint8)
        indxs = np.argmax(targets, axis=1)
        class_names = [class_names[i] for i in indxs]
        
        for i in range(num_imgs):
            full_img[:,i*img_w:(i+1)*img_w,:] = data[0][i,:,:,:]
            cv2.putText(full_img, class_names[i], (img_w*i + 10, 20), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        plt.figure(figsize = (20,2))
        plt.imshow(full_img)
        plt.show()

    
def plot_batch(data, targets):
    num_imgs = data[0].shape[0]
    it_val = len(data)
    fig, axs = plt.subplots(num_imgs, it_val, figsize=(
        30, 50), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)

    axs = axs.ravel()
    i = 0
    for img_idx, targ in zip(range(num_imgs), targets):
        for j in range(it_val):
            img = cv2.cvtColor(data[j][img_idx].astype(
                np.uint8), cv2.COLOR_BGR2RGB)
            axs[i+j].imshow(img)
            axs[i+j].set_title(targ)
        i += it_val

    plt.show()

def parse_net_params(filename='configs/road_signs.yml'):
    params = {}

    with open(filename, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    params_model = cfg['MODEL']
    params_train = cfg['TRAIN']
    params_paths = cfg['PATHS']

    if 'augmentations_type' in cfg:
        augmentations = get_aug(cfg['augmentation_type'], cfg['input_shape'])
    else:
        augmentations = None

    train_generator_parameters = {'dataset_path' : params_paths['dataset_path'],
                                'input_shape' : params_model['input_shape'],
                                'batch_size' : params_train['batch_size'],
                                'n_batches' : params_train['n_batches'],
                                'csv_file' : params_paths['csv_file'],
                                'image_id_column' : params_paths['image_id'],
                                'label_column' : params_paths['label'], 
                                'augmentations' : augmentations}

    learning_rate = cfg['learning_rate']
    if cfg['optimizer'] == 'adam':
        optimizer = optimizers.Adam(lr=learning_rate)
    elif cfg['optimizer'] == 'rms_prop':
        optimizer = optimizers.RMSprop(lr=learning_rate)
    elif cfg['optimizer'] == 'radam':
        from keras_radam import RAdam
        optimizer = RAdam(learning_rate)
    else:
        optimizer = optimizers.SGD(lr=learning_rate)


    params = {k: v for k, v in cfg.items() if k not in ['optimizer']}

    if 'dataset_path' in cfg:
        params['loader'] = EmbeddingNetImageLoader(cfg['dataset_path'],
                                                   input_shape=cfg['input_shape'],
                                                   min_n_obj_per_class=cfg['min_n_obj_per_class'],
                                                   select_max_n_obj_per_class = cfg['select_max_n_obj_per_class'], 
                                                   max_n_obj_per_class=cfg['max_n_obj_per_class'],
                                                   augmentations=augmentations)

    params['optimizer'] = optimizer
    return params
