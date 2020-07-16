import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import numpy as np
from embedding_net.models import EmbeddingNet, TripletNet, SiameseNet
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from embedding_net.datagenerators import ENDataLoader, SimpleDataGenerator, TripletsDataGenerator, SimpleTripletsDataGenerator, SiameseDataGenerator
from embedding_net.utils import parse_params, plot_grapths
from embedding_net.backbones import pretrain_backbone_softmax
from embedding_net.losses_and_accuracies import contrastive_loss, triplet_loss, accuracy
import argparse
from tensorflow import keras
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf




def parse_args():
    parser = argparse.ArgumentParser(description='Train a classificator')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')

    args = parser.parse_args()

    return args

def create_save_folders(params):
    work_dir_path = os.path.join(params['work_dir'], params['project_name'])
    weights_save_path = os.path.join(work_dir_path, 'weights/')
    weights_pretrained_save_path = os.path.join(work_dir_path, 'pretraining_model/weights/')
    encodings_save_path = os.path.join(work_dir_path, 'encodings/')
    plots_save_path = os.path.join(work_dir_path, 'plots/')
    tensorboard_save_path = os.path.join(work_dir_path, 'tf_log/')
    tensorboard_pretrained_save_path = os.path.join(work_dir_path, 'pretraining_model/tf_log/')
    weights_save_file_path = os.path.join(weights_save_path, 'epoch_{epoch:03d}' + '.hdf5')

    os.makedirs(work_dir_path , exist_ok=True)
    os.makedirs(weights_save_path, exist_ok=True)
    os.makedirs(weights_pretrained_save_path, exist_ok=True)
    os.makedirs(encodings_save_path, exist_ok=True)
    os.makedirs(plots_save_path, exist_ok=True)
    os.makedirs(tensorboard_pretrained_save_path, exist_ok=True)

    return tensorboard_save_path, weights_save_file_path, plots_save_path

def main():
    print('LOAD PARAMETERS')
    args = parse_args()
    cfg_params = parse_params(args.config)
    params_train = cfg_params['train']
    params_model = cfg_params['model']
    params_dataloader = cfg_params['dataloader']
    params_generator = cfg_params['generator']

    tensorboard_save_path, weights_save_file_path, plots_save_path = create_save_folders(cfg_params['general'])


    work_dir_path = os.path.join(cfg_params['general']['work_dir'],
                                 cfg_params['general']['project_name'])
    weights_save_path = os.path.join(work_dir_path, 'weights/')
    

    initial_lr = params_train['learning_rate']
    decay_factor = params_train['decay_factor']
    step_size = params_train['step_size']

    if params_dataloader['validate']:
        callback_monitor = 'val_loss'
    else:
        callback_monitor = 'loss'

    print('LOADING COMPLETED')
    callbacks = [
        LearningRateScheduler(lambda x: initial_lr *
                              decay_factor ** np.floor(x/step_size)),
        ReduceLROnPlateau(monitor=callback_monitor, factor=0.1,
                          patience=4, verbose=1),
        EarlyStopping(monitor=callback_monitor,
                      patience=10, 
                      verbose=1),
        ModelCheckpoint(filepath=weights_save_file_path,
                        monitor=callback_monitor, 
                        save_best_only=True,
                        verbose=1)
    ]
    
    print('CREATE DATALOADER')
    data_loader = ENDataLoader(**params_dataloader)
    print('DATALOADER CREATED!')

    if cfg_params['general']['tensorboard_callback']:
        callbacks.append(TensorBoard(log_dir=tensorboard_save_path))

    if cfg_params['general']['wandb_callback']:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init() 
        callbacks.append(WandbCallback(data_type="image", labels=data_loader.class_names))

    val_generator = None
    print('CREATE MODEL AND DATA GENETATORS')
    if params_model['mode'] == 'siamese':
        model = SiameseNet(cfg_params, training=True)
        train_generator = SiameseDataGenerator(class_files_paths=data_loader.train_data,
                                               class_names=data_loader.class_names,
                                               **params_generator)
        if data_loader.validate:
            val_generator = SiameseDataGenerator(class_files_paths=data_loader.val_data,
                                               class_names=data_loader.class_names,
                                               val_gen = True,
                                               **params_generator)
        losses = {'output_siamese' : contrastive_loss}
        metric = {'output_siamese' : accuracy}
    else:
        if cfg_params['general']['gpu_ids']:
            print('Multiple gpu mode')
            gpu_ids = cfg_params['general']['gpu_ids']
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            print(f'Using gpu ids: {gpu_ids}')
            gpu_ids_list = gpu_ids.split(',')
            n_gpu = len(gpu_ids_list)
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            n_gpu = 1
            print('Use single gpu mode')
        
        model = TripletNet(cfg_params, training=True)
        if n_gpu>1:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model.base_model = multi_gpu_model(model.base_model, gpus=n_gpu)
            # model.base_model = tf.keras.utils.multi_gpu_model(model.base_model, gpus=n_gpu)

        train_generator = TripletsDataGenerator(embedding_model=model.base_model,
                                            class_files_paths=data_loader.train_data,
                                            class_names=data_loader.class_names,
                                            **params_generator)

        if data_loader.validate:
            val_generator = SimpleTripletsDataGenerator(data_loader.val_data,
                                                    data_loader.class_names,
                                                    **params_generator)
        losses = triplet_loss(params_generator['margin'])
        metric = ['accuracy']
    print('DONE')


    if args.resume_from is not None:
        model.load_model(args.resume_from)
    
    print('COMPILE MODEL')
    model.model.compile(loss=losses, 
                        optimizer=params_train['optimizer'], 
                        metrics=metric)

    if 'softmax' in cfg_params:
        params_softmax = cfg_params['softmax']
        params_save_paths = cfg_params['general']
        pretrain_backbone_softmax(model.backbone_model, 
                                  data_loader, 
                                  params_softmax,  
                                  params_save_paths)

    history = model.model.fit_generator(train_generator,
                                validation_data=val_generator,  
                                epochs=params_train['n_epochs'], 
                                callbacks=callbacks,
                                verbose=1,
                                use_multiprocessing=False)

    if params_train['plot_history']:
        plot_grapths(history, plots_save_path)

if __name__ == '__main__':
    main()
