import os
import numpy as np
from embedding_net.model_new import EmbeddingNet, TripletNet
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from embedding_net.utils import parse_params, plot_grapths
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classificator')
    parser.add_argument('config', help='model config file path')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')

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
    weights_save_file_path = os.path.join(weights_save_path, 'best_' + params['project_name'] + '.h5')

    os.makedirs(work_dir_path , exist_ok=True)
    os.makedirs(weights_save_path, exist_ok=True)
    os.makedirs(weights_pretrained_save_path, exist_ok=True)
    os.makedirs(encodings_save_path, exist_ok=True)
    os.makedirs(plots_save_path, exist_ok=True)
    os.makedirs(tensorboard_pretrained_save_path, exist_ok=True)

    return tensorboard_save_path, weights_save_file_path, plots_save_path

def main():
    args = parse_args()
    cfg_params = parse_params(args.config)
    params_train = cfg_params['train']
    params_dataloader = cfg_params['dataloader']

    tensorboard_save_path, weights_save_file_path, plots_save_path = create_save_folders(cfg_params['save_paths'])

    model = TripletNet(cfg_params, training=True)

    if 'softmax' in cfg_params:
        model.pretrain_backbone_softmax()
    
    if args.resume_from is not None:
        model.load_model(args.resume_from)


    initial_lr = params_train['learning_rate']
    decay_factor = params_train['decay_factor']
    step_size = params_train['step_size']

    if params_dataloader['validate']:
        callback_monitor = 'val_loss'
    else:
        callback_monitor = 'loss'

    callbacks = [
        LearningRateScheduler(lambda x: initial_lr *
                              decay_factor ** np.floor(x/step_size)),
        ReduceLROnPlateau(monitor=callback_monitor, factor=0.1,
                          patience=4, verbose=1),
        EarlyStopping(monitor=callback_monitor,
                      patience=10, 
                      verbose=1),
        TensorBoard(log_dir=tensorboard_save_path),
        ModelCheckpoint(filepath=weights_save_file_path,
                        verbose=1, monitor=callback_monitor, save_best_only=True)
    ]

    history = model.train(callbacks=callbacks)

    if params_train['plot_history']:
        plot_grapths(history, plots_save_path)

    # if cfg_params['save_encodings']:
    #     encodings_save_file = os.path.join(
    #         encodings_save_path, cfg_params['encodings_save_name'])
    #     model.generate_encodings(save_file_name=encodings_save_file,
    #                              max_num_samples_of_each_class=cfg_params['max_num_samples_of_each_class'],
    #                              knn_k=cfg_params['knn_k'],
    #                              shuffle=True)
    #     if cfg_params['to_validate']:
    #         model_accuracies = model.calculate_prediction_accuracy()
    #         print('Model top1 accuracy on validation set: {}'.format(model_accuracies['top1']))
    #         print('Model top5 accuracy on validation set: {}'.format(model_accuracies['top5']))


if __name__ == '__main__':
    main()
