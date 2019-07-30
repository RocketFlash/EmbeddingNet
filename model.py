import os
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2
from keras.utils import plot_model
from keras.layers import Dense, Input, Lambda, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from classification_models import Classifiers
from keras.callbacks import TensorBoard


class SiameseNet:
    """
    SiameseNet for image classification
    mode = 'l1' -> l1_loss
    mode = 'l2' -> l2_loss

    """

    def __init__(self, input_shape, image_loader, mode='l1', backbone='resnet50', optimizer=optimizers.Adam(lr=1e-4), tensorboard_log_path='tf_log'):
        self.input_shape = input_shape
        self.backbone = backbone
        self.mode = mode
        self.optimizer = optimizer
        self.model = []
        self._create_model()
        self.data_loader = image_loader
        self.base_model = []
        if tensorboard_log_path:
            os.makedirs(tensorboard_log_path, exist_ok=True)
        self.tensorboard_callback = TensorBoard(
            tensorboard_log_path) if tensorboard_log_path else None
        if self.tensorboard_callback:
            self.tensorboard_callback.set_model(self.model)

    def _create_model(self):

        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        if self.backbone == 'simple':
            input_image = Input(self.input_shape)
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
            encoded_output = Dense(2048, activation='sigmoid',
                                   kernel_regularizer=l2(1e-3))(x)
            self.base_model = Model(
                inputs=[input_image], outputs=[encoded_output])
        else:
            classifier, preprocess_input = Classifiers.get(self.backbone)
            backbone_model = classifier(
                input_shape=self.input_shape, weights='imagenet', include_top=False)

            for layer in backbone_model.layers:
                layer.trainable = False

            after_backbone = backbone_model.output
            x = Flatten()(after_backbone)
            x = Dense(512, activation="relu")(x)
            x = Dropout(0.1)(x)
            x = Dense(256, activation="relu")(x)
            x = Dropout(0.1)(x)
            encoded_output = Dense(128, activation="relu")(x)

            self.base_model = Model(
                inputs=[backbone_model.input], outputs=[encoded_output])

        image_encoding_1 = self.base_model(input_image_1)
        image_encoding_2 = self.base_model(input_image_2)

        if self.mode == 'l1':
            L1_layer = Lambda(
                lambda tensors: K.abs(tensors[0] - tensors[1]))
            distance = L1_layer([image_encoding_1, image_encoding_2])

            prediction = Dense(units=1, activation='sigmoid')(distance)
            metric = 'binary_accuracy'
        elif self.mode == 'l2':

            L2_layer = Lambda(
                lambda tensors: K.sqrt(K.maximum(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True), K.epsilon())))
            distance = L2_layer([image_encoding_1, image_encoding_2])

            prediction = distance
            metric = self.accuracy
        self.model = Model(
            inputs=[input_image_1, input_image_2], outputs=prediction)

        plot_model(self.model, to_file='plots/model.png')
        print('BASE MODEL SUMMARY')
        self.base_model.summary()

        print('WHOLE MODEL SUMMARY')
        self.model.summary()
        self.model.compile(loss=self.contrastive_loss, metrics=[metric],
                           optimizer=self.optimizer)

    def write_log(self, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.tensorboard_callback.writer.add_summary(summary, batch_no)
            self.tensorboard_callback.writer.flush()

    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        sqaure_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    def train_on_batch(self, batch_size=8, s="train"):
        generator = self.data_loader.generate(batch_size, s=s)
        pairs, targets = next(generator)
        train_loss, train_accuracy = self.model.train_on_batch(
            pairs, targets)
        return train_loss, train_accuracy

    def validate_on_batch(self, batch_size=8, s="val"):
        generator = self.data_loader.generate(batch_size, s=s)
        pairs, targets = next(generator)
        val_loss, val_accuracy = self.model.test_on_batch(
            pairs, targets)
        return val_loss, val_accuracy

    def train(self, steps_per_epoch, epochs, with_val=True, batch_size=8, verbose=1):
        generator_train = self.data_loader.generate(batch_size, 'train')
        train_accuracies_epochs = []
        train_losses_epochs = []
        val_accuracies_epochs = []
        val_losses_epochs = []
        tensorboard_names = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        for j in range(epochs):
            train_accuracies_it = []
            train_losses_it = []
            for i in range(steps_per_epoch):
                pairs, targets = next(generator_train)
                train_loss_it, train_accuracy_it = self.model.train_on_batch(
                    pairs, targets)
                train_accuracies_it.append(train_accuracy_it)
                train_losses_it.append(train_loss_it)
            train_loss_epoch = sum(train_losses_it) / len(train_losses_it)
            train_accuracy_epoch = sum(
                train_accuracies_it) / len(train_accuracies_it)
            train_accuracies_epochs.append(train_accuracy_epoch)
            train_losses_epochs.append(train_loss_epoch)

            if with_val:
                val_loss, val_accuracy = self.validate()
                val_accuracies_epochs.append(val_accuracy)
                val_losses_epochs.append(val_loss)
                if verbose:
                    print('[Epoch {}] train_loss: {} , train_acc: {}, val_loss: {} , val_acc: {}'.format(
                        j, train_loss_epoch, train_accuracy_epoch, val_loss, val_accuracy))
                logs = [train_loss_epoch, train_accuracy_epoch,
                        val_loss, val_accuracy]
            else:
                if verbose:
                    print('[Epoch {}] train_loss: {} , train_acc: {}'.format(
                        j, train_loss_epoch, train_accuracy_epoch))
                tensorboard_names = tensorboard_names[:2]
                logs = [train_loss_epoch, train_accuracy_epoch]
            if self.tensorboard_callback:
                self.write_log(tensorboard_names, logs, j)
        if with_val:
            return train_losses_epochs, train_accuracies_epochs, val_losses_epochs, val_accuracies_epochs
        else:
            return train_losses_epochs, train_accuracies_epochs

    def validate(self, number_of_comparisons=100, batch_size=4, s="val"):
        generator = self.data_loader.generate(batch_size, s=s)
        val_accuracies_it = []
        val_losses_it = []
        for _ in range(number_of_comparisons):
            pairs, targets = next(generator)
            val_loss_it, val_accuracy_it = self.model.test_on_batch(
                pairs, targets)
            val_accuracies_it.append(val_accuracy_it)
            val_losses_it.append(val_loss_it)
        val_loss_epoch = sum(val_losses_it) / len(val_losses_it)
        val_accuracy_epoch = sum(
            val_accuracies_it) / len(val_accuracies_it)
        return val_loss_epoch, val_accuracy_epoch

    def generate_encodings(self):

        paths = self.data_loader.images_paths['train']
        labels = self.data_loader.images_labels['train']

        data = {}
        for path in paths:
            info = {}
            img = cv2.imread(path)
            info['encoding'] = self.base_model.predict(img)

    def predict(self, batch_size=8):
        pass
