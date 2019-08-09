import os
import glob
import numpy as np
import keras.backend as K
import tensorflow as tf
import pickle
import cv2
import random
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2
from keras.utils import plot_model
from keras.layers import Dense, Input, Lambda, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from classification_models import Classifiers



class SiameseNet:
    """
    SiameseNet for image classification
    mode = 'l1' -> l1_loss
    mode = 'l2' -> l2_loss

    """

    def __init__(self, input_shape, image_loader, mode='l1', backbone='resnet50',
                 backbone_weights = 'imagenet',
                 optimizer=optimizers.Adam(lr=1e-4), tensorboard_log_path='tf_log/',
                 weights_save_path='weights/', plots_path='plots/', encodings_path='encodings/',
                 project_name='', freeze_backbone=True):
        self.input_shape = input_shape
        self.backbone = backbone
        self.backbone_weights = backbone_weights
        self.mode = mode
        self.project_name = project_name
        self.optimizer = optimizer
        self.model = []
        self.base_model = []
        self.l_model = []
        self.freeze_backbone = freeze_backbone

        self.encodings_path = os.path.join(encodings_path, project_name)
        os.makedirs(self.encodings_path, exist_ok=True)
        self.plots_path = os.path.join(plots_path, project_name)
        self.tensorboard_log_path = os.path.join(
            tensorboard_log_path, project_name)
        if self.plots_path:
            os.makedirs(self.plots_path, exist_ok=True)
        if self.tensorboard_log_path:
            os.makedirs(self.tensorboard_log_path, exist_ok=True)
        self.tensorboard_callback = TensorBoard(
            self.tensorboard_log_path) if tensorboard_log_path else None
        if self.tensorboard_callback:
            events_files_list = glob.glob(
                os.path.join(self.tensorboard_log_path, 'events*'))
            for event_file in events_files_list:
                try:
                    os.remove(event_file)
                except:
                    print("Error while deleting file : ", event_file)
            self.tensorboard_callback.set_model(self.model)
        self.weights_save_path = os.path.join(
            weights_save_path, self.project_name)
        if self.weights_save_path:
            os.makedirs(self.weights_save_path, exist_ok=True)

        self._create_model()
        self.data_loader = image_loader
        self.encoded_training_data = {}

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
            encoded_output = Dense(4096, activation='sigmoid',
                                   kernel_regularizer=l2(1e-3))(x)
            self.base_model = Model(
                inputs=[input_image], outputs=[encoded_output])
        else:
            classifier, preprocess_input = Classifiers.get(self.backbone)
            backbone_model = classifier(
                input_shape=self.input_shape, weights=self.backbone_weights, include_top=False)

            if self.freeze_backbone:
                for layer in backbone_model.layers:
                    layer.trainable = False

            after_backbone = backbone_model.output
            x = Flatten()(after_backbone)
            # x = Dense(512, activation="relu")(x)
            # x = Dropout(0.5)(x)
            # x = Dense(512, activation="relu")(x)
            # x = Dropout(0.5)(x)
            encoded_output = Dense(4096, activation="relu")(x)

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

        # self.l_model = Model(inputs=[image_encoding_1, image_encoding_2], outputs=[prediction])
        self.model = Model(
            inputs=[input_image_1, input_image_2], outputs=prediction)

        plot_model(self.model, to_file='{}model.png'.format(self.plots_path))
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

    def train_generator(self, steps_per_epoch, epochs, callbacks = [], val_steps=100, with_val=True, batch_size=8, verbose=1):

        train_generator = self.data_loader.generate(batch_size, s="train")
        val_generator = self.data_loader.generate(batch_size, s="val")
        
        history = self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                 verbose=verbose, validation_data = val_generator, validation_steps = val_steps, callbacks=callbacks)
        return history

    def validate(self, number_of_comparisons=100, batch_size=4, s="val"):
        generator = self.data_loader.generate(batch_size, s=s)
        val_accuracies_it = []
        val_losses_it = []
        for _ in range(number_of_comparisons):
            pairs, targets = next(generator)
            # predictions = self.model.predict(pairs)

            val_loss_it, val_accuracy_it = self.model.test_on_batch(
                pairs, targets)
            # print(predictions)
            # print(targets)
            # print('================================')
            val_accuracies_it.append(val_accuracy_it)
            val_losses_it.append(val_loss_it)
        val_loss_epoch = sum(val_losses_it) / len(val_losses_it)
        val_accuracy_epoch = sum(
            val_accuracies_it) / len(val_accuracies_it)
        return val_loss_epoch, val_accuracy_epoch

    def _generate_encoding(self, img_path):
        img = self.data_loader.get_image(img_path)
        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        return encoding

    def generate_encodings(self, save_file_name='encodings.pkl', max_num_samples_of_each_classes=10, shuffle = True):
        data_paths, data_labels, data_encodings = [], [], []
        classes_counter = {}

        if shuffle:
            c = list(zip(self.data_loader.images_paths['train'], self.data_loader.images_labels['train']))
            random.shuffle(c)
            self.data_loader.images_paths['train'], self.data_loader.images_labels['train'] = zip(*c)

        for img_path, img_label in zip(self.data_loader.images_paths['train'],
                                       self.data_loader.images_labels['train']):
            if img_label not in classes_counter:
                classes_counter[img_label] = 0
            classes_counter[img_label] += 1
            if classes_counter[img_label] < max_num_samples_of_each_classes:
                data_paths.append(img_path)
                data_labels.append(img_label)
                data_encodings.append(self._generate_encoding(img_path))
        self.encoded_training_data['paths'] = data_paths
        self.encoded_training_data['labels'] = data_labels
        self.encoded_training_data['encodings'] = np.squeeze(
            np.array(data_encodings))

        f = open(os.path.join(self.encodings_path, save_file_name), "wb")
        pickle.dump(self.encoded_training_data, f)
        f.close()

    def load_encodings(self, path_to_encodings):
        try:
            with open(path_to_encodings, 'rb') as f:
                self.encoded_training_data = pickle.load(f)
        except:
            print("Problem with encodings file")

    def calculate_distances(self, encoding):
        training_encodings = self.encoded_training_data['encodings']
        return np.sqrt(
            np.sum((training_encodings - np.array(encoding))**2, axis=1))

    def predict(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        distances = self.calculate_distances(encoding)
        max_element = np.argmin(distances)
        predicted_label = self.encoded_training_data['labels'][max_element]
        return predicted_label

    def calculate_prediction_accuracy(self):
        pass