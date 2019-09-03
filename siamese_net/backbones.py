from keras.layers import Dense, Input, Lambda, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, concatenate
from classification_models import Classifiers
from keras.models import Model
from keras.regularizers import l2

def get_backbone(input_shape,encodings_len=4096,backbone_type='simple',backbone_weights='imagenet',freeze_backbone=False):
        if backbone_type == 'simple':
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
            encoded_output = Dense(encodings_len, activation='sigmoid',
                                   kernel_regularizer=l2(1e-3))(x)
            base_model = Model(
                inputs=[input_image], outputs=[encoded_output])
        elif backbone_type == 'simple2':
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
            x = Flatten()(x)
            x = Dense(512, activation="relu")(x)
            x = Dropout(0.5)(x)
            encoded_output = Dense(encodings_len, activation='sigmoid',
                                   kernel_regularizer=l2(1e-3))(x)
            base_model = Model(
                inputs=[input_image], outputs=[encoded_output])
        else:
            classifier, preprocess_input = Classifiers.get(backbone_type)
            backbone_model = classifier(input_shape=input_shape, 
                                        weights=backbone_weights, 
                                        include_top=False)

            if freeze_backbone:
                for layer in backbone_model.layers[:-2]:
                    layer.trainable = False

            after_backbone = backbone_model.output
            x = Flatten()(after_backbone)
            # x = Dense(512, activation="relu")(x)
            # x = Dropout(0.5)(x)
            # x = Dense(512, activation="relu")(x)
            # x = Dropout(0.5)(x)
            encoded_output = Dense(encodings_len, activation="relu")(x)

            base_model = Model(
                inputs=[backbone_model.input], outputs=[encoded_output])

        return base_model
