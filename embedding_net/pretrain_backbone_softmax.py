import keras
import numpy as np
from classification_models import Classifiers
from .data_loader import SimpleNetImageLoader
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def pretrain_backbone_softmax(input_model, cfg_params):

    input_shape = cfg_params['input_shape']
    dataset_path = cfg_params['dataset_path']
    image_loader = SimpleNetImageLoader(
        dataset_path, input_shape=input_shape, augmentations=None)
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

    train_generator = image_loader.generate(batch_size, s="train")
    val_generator = image_loader.generate(batch_size, s="val")

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                          patience=4, verbose=1),
        EarlyStopping(patience=50, verbose=1),
        TensorBoard(log_dir='tf_log/')
    ]

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=val_generator,
                                  validation_steps=val_steps,
                                  callbacks=callbacks)

    return input_model
