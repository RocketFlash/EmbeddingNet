import keras
import os
import numpy as np
from classification_models.keras import Classifiers
from .data_loader import EmbeddingNetImageLoader
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def pretrain_backbone_softmax(input_model, cfg_params):

    input_shape = cfg_params['input_shape']
    dataset_path = cfg_params['dataset_path']
    image_loader = EmbeddingNetImageLoader(cfg['dataset_path'],
                                           input_shape=cfg['input_shape'],
                                           min_n_obj_per_class=cfg['min_n_obj_per_class'], 
                                           max_n_obj_per_class=cfg['max_n_obj_per_class'],
                                           augmentations=None)
    n_classes = image_loader.n_classes['train']

    x = keras.layers.GlobalAveragePooling2D()(input_model.output)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[input_model.input], outputs=[output])

    # train
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    batch_size = cfg_params['softmax_batch_size']
    val_steps = cfg_params['softmax_val_steps']
    steps_per_epoch = cfg_params['softmax_steps_per_epoch']
    epochs = cfg_params['softmax_epochs']

    train_generator = image_loader.generate(batch_size, mode='simple', s="train")
    val_generator = image_loader.generate(batch_size, mode='simple', s="val")

    tensorboard_save_path = os.path.join(
        cfg_params['work_dir'], 'tf_log/pretraining_model/')
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                          patience=4, verbose=1),
        TensorBoard(log_dir=tensorboard_save_path)
    ]

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=val_generator,
                                  validation_steps=val_steps,
                                  callbacks=callbacks)

    return input_model
