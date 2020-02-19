from tensorflow.keras.layers import Dense, Input, Lambda, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


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
        print(backbone_model.output)
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

    return base_model, backbone_model
