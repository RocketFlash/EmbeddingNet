import os
import numpy as np
from embedding_net.model import EmbeddingNet
from embedding_net.pretrain_backbone_softmax import pretrain_backbone_softmax
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from embedding_net.utils import parse_net_params, plot_grapths
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classificator')
    parser.add_argument('config', help='model config file path')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_params = parse_net_params(args.config)
    os.makedirs(cfg_params['work_dir'], exist_ok=True)
    weights_save_path = os.path.join(cfg_params['work_dir'], 'weights/')
    encodings_save_path = os.path.join(cfg_params['work_dir'], 'encodings/')
    plots_save_path = os.path.join(cfg_params['work_dir'], 'plots/')
    tensorboard_save_path = os.path.join(cfg_params['work_dir'], 'tf_log/')

    os.makedirs(weights_save_path, exist_ok=True)
    os.makedirs(encodings_save_path, exist_ok=True)

    model = EmbeddingNet(cfg_params)

    weights_save_file = os.path.join(
        weights_save_path, cfg_params['model_save_name'])

    initial_lr = cfg_params['learning_rate']
    decay_factor = cfg_params['decay_factor']
    step_size = cfg_params['step_size']

    callbacks = [
        LearningRateScheduler(lambda x: initial_lr *
                              decay_factor ** np.floor(x/step_size)),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                          patience=4, verbose=1),
        EarlyStopping(patience=5, verbose=1),
        TensorBoard(log_dir=tensorboard_save_path),
        ModelCheckpoint(filepath=weights_save_file,
                        verbose=1, monitor='val_loss', save_best_only=True)
    ]

    history = model.train_generator_mining(steps_per_epoch=cfg_params['n_steps_per_epoch'],
                                           epochs=cfg_params['n_epochs'],
                                           callbacks=callbacks,
                                           val_steps=cfg_params['val_steps'],
                                           val_batch=cfg_params['val_batch_size'],
                                           n_classes=cfg_params['mining_n_classes'],
                                           n_samples=cfg_params['mining_n_samples'],
                                           negative_selection_mode=cfg_params['negatives_selection_mode'])

    if cfg_params['plot_history']:
        os.makedirs(plots_save_path, exist_ok=True)
        plot_grapths(history, plots_save_path)

    if cfg_params['save_encodings']:
        encodings_save_file = os.path.join(
            encodings_save_path, cfg_params['encodings_save_name'])
        model.generate_encodings(save_file_name=encodings_save_file,
                                 max_num_samples_of_each_class=cfg_params['max_num_samples_of_each_class'],
                                 knn_k=cfg_params['knn_k'],
                                 shuffle=True)

        model_accuracy = model.calculate_prediction_accuracy()
        print('Model accuracy on validation set: {}'.format(model_accuracy))


if __name__ == '__main__':
    main()
