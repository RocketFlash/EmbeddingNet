from sklearn.manifold import TSNE
import os
os.environ["TF_KERAS"] = '1'
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
import yaml
from tensorflow.keras import optimizers
from .augmentations import get_aug


def get_image(img_path, input_shape=None):
    img = cv2.imread(img_path)
    if img is None:
        print('image is not exist ' + img_path)
        return None
    if input_shape:
        img = cv2.resize(
            img, (input_shape[0], input_shape[1]))
    return img

def get_images(img_paths, input_shape=None):
    imgs = [get_image(img_path, input_shape) for img_path in img_paths]
    return np.array(imgs)



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


def plot_tsne_interactive(encodings):
    import plotly.graph_objects as go
    if type(encodings) is str:
        encodings = load_encodings(encodings)
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
                                 text=str(l),
                                 name=str(l)))
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
            full_img[:,i*img_w:(i+1)*img_w,:] = data[0][i,:,:,::-1]*255
            cv2.putText(full_img, class_names[i], (img_w*i + 5, 20), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.2, (0, 255, 0), 1, cv2.LINE_AA)
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
            image = data[j][img_idx]*255
            img = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            axs[i+j].imshow(img)
            # axs[i+j].set_title(targ)
        i += it_val

    plt.show()


def get_optimizer(name, learning_rate):
    if name == 'adam':
        optimizer = optimizers.Adam(lr=learning_rate)
    elif name == 'rms_prop':
        optimizer = optimizers.RMSprop(lr=learning_rate)
    elif name == 'radam':
        from keras_radam import RAdam
        optimizer = RAdam(learning_rate)
    else:
        optimizer = optimizers.SGD(lr=learning_rate)
    return optimizer


def parse_params(filename='configs/road_signs.yml'):
    with open(filename, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    if 'augmentations_type' in cfg['GENERATOR']:
        augmentations = get_aug(cfg['GENERATOR']['augmentation_type'], 
                                cfg['MODEL']['input_shape'])
    else:
        augmentations = None

    optimizer = get_optimizer(cfg['TRAIN']['optimizer'], 
                              cfg['TRAIN']['learning_rate'])

    params_dataloader = cfg['DATALOADER']
    params_generator = cfg['GENERATOR']
    params_model = cfg['MODEL']
    params_train = cfg['TRAIN']
    params_general = cfg['GENERAL']
    params_encodings = cfg['ENCODINGS']

    params_generator['input_shape'] = params_model['input_shape']
    params_train['optimizer'] = optimizer
    params_generator['augmentations'] = augmentations

    params = {'dataloader' : params_dataloader,
              'generator' : params_generator,
              'model' : params_model,
              'train' : params_train,
              'general': params_general,
              'encodings' : params_encodings}

    if 'SOFTMAX_PRETRAINING' in cfg:
        params_softmax = cfg['SOFTMAX_PRETRAINING']
        params_softmax['augmentations'] = augmentations
        params_softmax['input_shape'] = params_model['input_shape']
        softmax_optimizer = get_optimizer(cfg['SOFTMAX_PRETRAINING']['optimizer'], 
                              cfg['SOFTMAX_PRETRAINING']['learning_rate'])
        params_softmax['optimizer'] = softmax_optimizer
        params['softmax'] =  params_softmax
        

    return params
