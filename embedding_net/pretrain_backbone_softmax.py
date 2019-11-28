import keras
import numpy as np
import yaml
from classification_models import Classifiers
from .data_loader import SimpleNetImageLoader
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



def pretrain_backbone_softmax(input_model, config_file):

    backbone_model = input_model.backbone_model
    with open(config_file, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    input_shape = cfg['input_shape']
    dataset_path = cfg['dataset_path']
    image_loader = SimpleNetImageLoader(dataset_path, input_shape=input_shape, augmentations = None)
    n_classes = image_loader.n_classes['train']

    x = keras.layers.GlobalAveragePooling2D()(backbone_model.output)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[backbone_model.input], outputs=[output])

    # train
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    batch_size = 8
    val_steps = 200
    steps_per_epoch = 500
    epochs = 20
    train_generator = image_loader.generate(batch_size, s="train")
    val_generator = image_loader.generate(batch_size, s="val")

    initial_lr = 1e-4
    decay_factor = 0.95
    step_size = 1

    callbacks = [
        LearningRateScheduler(lambda x: initial_lr *
                            decay_factor ** np.floor(x/step_size)),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1),
        EarlyStopping(patience=50, verbose=1),
        TensorBoard(log_dir='tf_log/')
    ]

    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                    verbose=1, validation_data = val_generator, validation_steps = val_steps, callbacks=callbacks)

    return backbone_model