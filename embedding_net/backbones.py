from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from .datagenerators import SimpleDataGenerator
import os
import numpy as np

def get_backbone(input_shape,
                 encodings_len=4096,
                 backbone_name='simple',
                 embeddings_normalization=True,
                 backbone_weights='imagenet',
                 freeze_backbone=False,
                 **kwargs):
    if backbone_name == 'simple':
        input_image = Input(input_shape)
        x = Conv2D(64, (10, 10), activation='relu',
                   kernel_regularizer=l2(2e-4))(input_image)
        x = MaxPool2D()(x)
        x = Conv2D(128, (7, 7), activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = MaxPool2D()(x)
        x = Conv2D(128, (4, 4), activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = MaxPool2D()(x)
        x = Conv2D(256, (4, 4), activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = Flatten()(x)
        backbone_model = Model(
            inputs=[input_image], outputs=[x])
        encoded_output = Dense(encodings_len, activation='relu',
                               kernel_regularizer=l2(1e-3))(x)
        if embeddings_normalization:
            encoded_output = Lambda(lambda x: K.l2_normalize(
                x, axis=1), name='l2_norm')(encoded_output)
        base_model = Model(
            inputs=[input_image], outputs=[encoded_output])
    elif backbone_name == 'simple2':
        input_image = Input(input_shape)
        x = Conv2D(32, kernel_size=3, activation='relu',
                   kernel_regularizer=l2(2e-4))(input_image)
        x = BatchNormalization()(x)
        x = Conv2D(32, kernel_size=3, activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Conv2D(64, kernel_size=3, activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=3, activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Conv2D(128, kernel_size=4, activation='relu',
                   kernel_regularizer=l2(2e-4))(x)
        x = BatchNormalization()(x)
        backbone_model = Model(
            inputs=[input_image], outputs=[x])
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        encoded_output = Dense(encodings_len, activation='relu',
                               kernel_regularizer=l2(1e-3))(x)
        if embeddings_normalization:
            encoded_output = Lambda(lambda x: K.l2_normalize(
                x, axis=1), name='l2_norm')(encoded_output)

        base_model = Model(
            inputs=[input_image], outputs=[encoded_output])
    else:
        if backbone_name.startswith('efficientnet'):
            import efficientnet.tfkeras as efn
            efficientnet_models = {
                'efficientnet-b0': efn.EfficientNetB0,
                'efficientnet-b1': efn.EfficientNetB1,
                'efficientnet-b2': efn.EfficientNetB2,
                'efficientnet-b3': efn.EfficientNetB3,
                'efficientnet-b4': efn.EfficientNetB4,
                'efficientnet-b5': efn.EfficientNetB5,
                'efficientnet-b6': efn.EfficientNetB6,
                'efficientnet-b7': efn.EfficientNetB7,
            }
            Efficientnet_model = efficientnet_models[backbone_name]
            backbone_model = Efficientnet_model(input_shape=input_shape, 
                                            weights=backbone_weights, 
                                            include_top=False)
        else:
            from classification_models.tfkeras import Classifiers
            classifier, preprocess_input = Classifiers.get(backbone_name)
            backbone_model = classifier(input_shape=input_shape,
                                        weights=backbone_weights,
                                        include_top=False)

        if freeze_backbone:
            for layer in backbone_model.layers[:-2]:
                layer.trainable = False
        
        after_backbone = backbone_model.output
        x = GlobalAveragePooling2D()(after_backbone)
        # x = Flatten()(after_backbone)

        x = Dense(encodings_len//2, activation="relu")(x)

        encoded_output = Dense(encodings_len, activation="relu")(x)
        if embeddings_normalization:
            encoded_output = Lambda(lambda x: K.l2_normalize(
                x, axis=1), name='l2_norm')(encoded_output)
        base_model = Model(
            inputs=[backbone_model.input], outputs=[encoded_output])

        # base_model._make_predict_function()

    return base_model, backbone_model


def pretrain_backbone_softmax(backbone_model, data_loader, params_softmax,  params_save_paths):

    optimizer = params_softmax['optimizer']
    learning_rate = params_softmax['learning_rate']
    decay_factor = params_softmax['decay_factor']
    step_size = params_softmax['step_size']

    input_shape = params_softmax['input_shape']
    batch_size = params_softmax['batch_size']
    val_steps = params_softmax['val_steps']
    steps_per_epoch = params_softmax['steps_per_epoch']
    n_epochs = params_softmax['n_epochs']
    augmentations = params_softmax['augmentations']

    n_classes = data_loader.n_classes

    x = GlobalAveragePooling2D()(backbone_model.output)

    output = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=[backbone_model.input], outputs=[output])

    # train
    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

    train_generator = SimpleDataGenerator(data_loader.train_data,
                                            data_loader.class_names,
                                            input_shape=input_shape,
                                            batch_size = batch_size,
                                            n_batches = steps_per_epoch, 
                                            augmentations=augmentations)

    if data_loader.validate:
        val_generator = SimpleDataGenerator(data_loader.val_data,
                                            data_loader.class_names,
                                            input_shape=input_shape,
                                            batch_size = batch_size,
                                            n_batches = steps_per_epoch, 
                                            augmentations=augmentations)
        checkpoint_callback_monitor = 'val_loss'
    else:
        val_generator = None
        checkpoint_callback_monitor = 'loss'

    tensorboard_save_path = os.path.join(
        params_save_paths['work_dir'],
        params_save_paths['project_name'], 
        'pretraining_model/tf_log/')
    weights_save_file = os.path.join(
        params_save_paths['work_dir'],
        params_save_paths['project_name'], 
        'pretraining_model/weights/',
        params_save_paths['project_name']+'_{epoch:03d}' +'.h5')

    callbacks = [
        LearningRateScheduler(lambda x: learning_rate *
                            decay_factor ** np.floor(x/step_size)),
        ReduceLROnPlateau(monitor=checkpoint_callback_monitor, factor=0.1,
                        patience=20, verbose=1),
        EarlyStopping(monitor=checkpoint_callback_monitor,
                        patience=10, 
                        verbose=1, 
                        restore_best_weights=True),
        # TensorBoard(log_dir=tensorboard_save_path),
        ModelCheckpoint(filepath=weights_save_file,
                        verbose=1, 
                        monitor=checkpoint_callback_monitor, 
                        save_best_only=True)]
                        
    history = model.fit_generator(train_generator,
                                steps_per_epoch=steps_per_epoch,
                                epochs=n_epochs,
                                verbose=1,
                                validation_data=val_generator,
                                validation_steps=val_steps,
                                callbacks=callbacks)