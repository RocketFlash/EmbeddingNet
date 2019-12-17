import os
import numpy as np
import keras.backend as K
import cv2
import random
from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Dense, Input, Lambda, concatenate
import pickle
from .utils import load_encodings
from .backbones import get_backbone
from .pretrain_backbone_softmax import pretrain_backbone_softmax
from . import losses_and_accuracies as lac
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# TODO
# [] - implement magnet loss
# [] - finalize settings with l1 and l2 losses


class EmbeddingNet:
    """
    SiameseNet for image classification
    distance_type = 'l1' -> l1_loss
    distance_type = 'l2' -> l2_loss

    mode = 'siamese' -> Siamese network
    mode = 'triplet' -> Triplen network
    """

    def __init__(self,  cfg_params):
        self.input_shape = cfg_params['input_shape']
        self.encodings_len = cfg_params['encodings_len']
        self.backbone = cfg_params['backbone']
        self.backbone_weights = cfg_params['backbone_weights']
        self.distance_type = cfg_params['distance_type']
        self.mode = cfg_params['mode']
        self.optimizer = cfg_params['optimizer']
        self.freeze_backbone = cfg_params['freeze_backbone']
        self.data_loader = cfg_params['loader']
        self.embeddings_normalization = cfg_params['embeddings_normalization']
        self.margin = cfg_params['margin']

        self.model = []
        self.base_model = []
        self.backbone_model = []

        if self.mode == 'siamese':
            self._create_model_siamese()
        elif self.mode == 'triplet':
            self._create_model_triplet()

        self.encoded_training_data = {}

        if cfg_params['softmax_pretraining']:
            pretrain_backbone_softmax(self.backbone_model, cfg_params)

    def _create_base_model(self):
        self.base_model, self.backbone_model = get_backbone(input_shape=self.input_shape,
                                                            encodings_len=self.encodings_len,
                                                            backbone_type=self.backbone,
                                                            embeddings_normalization=self.embeddings_normalization,
                                                            backbone_weights=self.backbone_weights,
                                                            freeze_backbone=self.freeze_backbone)

    def _create_model_siamese(self):

        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        self._create_base_model()
        self.base_model._make_predict_function()

        image_encoding_1 = self.base_model(input_image_1)
        image_encoding_2 = self.base_model(input_image_2)

        if self.distance_type == 'l1':
            L1_layer = Lambda(
                lambda tensors: K.abs(tensors[0] - tensors[1]))
            distance = L1_layer([image_encoding_1, image_encoding_2])

            prediction = Dense(units=1, activation='sigmoid')(distance)
            metric = 'binary_accuracy'

        elif self.distance_type == 'l2':

            L2_layer = Lambda(
                lambda tensors: K.sqrt(K.maximum(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True), K.epsilon())))
            distance = L2_layer([image_encoding_1, image_encoding_2])

            prediction = distance
            metric = lac.accuracy

        self.model = Model(
            inputs=[input_image_1, input_image_2], outputs=prediction)

        print('Base model summary')
        self.base_model.summary()

        print('Whole model summary')
        self.model.summary()

        self.model.compile(loss=lac.contrastive_loss, metrics=[metric],
                           optimizer=self.optimizer)

    def _create_model_triplet(self):
        input_image_a = Input(self.input_shape)
        input_image_p = Input(self.input_shape)
        input_image_n = Input(self.input_shape)

        self._create_base_model()
        self.base_model._make_predict_function()
        image_encoding_a = self.base_model(input_image_a)
        image_encoding_p = self.base_model(input_image_p)
        image_encoding_n = self.base_model(input_image_n)

        merged_vector = concatenate([image_encoding_a, image_encoding_p, image_encoding_n],
                                    axis=-1, name='merged_layer')
        self.model = Model(inputs=[input_image_a, input_image_p, input_image_n],
                           outputs=merged_vector)

        print('Base model summary')
        self.base_model.summary()

        print('Whole model summary')
        self.model.summary()

        self.model.compile(loss=lac.triplet_loss(
            self.margin), optimizer=self.optimizer)

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

    def train_generator(self, steps_per_epoch, epochs, callbacks=[], val_steps=100, with_val=True, batch_size=8, verbose=1):

        train_generator = self.data_loader.generate(
            batch_size, mode=self.mode, s="train")
        val_generator = self.data_loader.generate(
            batch_size, mode=self.mode, s="val")

        history = self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                           verbose=verbose, validation_data=val_generator, validation_steps=val_steps, callbacks=callbacks)

        return history

    def train_generator_mining(self,
                               steps_per_epoch,
                               epochs, callbacks=[],
                               val_steps=100,
                               with_val=True,
                               n_classes=4,
                               n_samples=4,
                               val_batch=8,
                               negative_selection_mode='semihard',
                               verbose=1):

        train_generator = self.data_loader.generate_mining(
            self.base_model, n_classes, n_samples, margin=self.margin, negative_selection_mode=negative_selection_mode, s="train")
        val_generator = self.data_loader.generate(
            val_batch, mode=self.mode, s="val")

        history = self.model.fit_generator(train_generator,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           verbose=verbose,
                                           validation_data=val_generator,
                                           validation_steps=val_steps,
                                           callbacks=callbacks)
        return history

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


    def _generate_encoding(self, img_path):
        img = self.data_loader.get_image(img_path)
        if img is None:
            return None
        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        return encoding


    def generate_encodings(self, save_file_name='encodings.pkl', only_centers=False, max_num_samples_of_each_class=10, knn_k=1, shuffle=True):
        data_paths, data_labels, data_encodings = [], [], []
        classes_counter = {}
        classes_encodings = {}
        k_val = 1 if only_centers else knn_k

        if shuffle:
            c = list(zip(
                self.data_loader.images_paths['train'], self.data_loader.images_labels['train']))
            random.shuffle(c)
            self.data_loader.images_paths['train'], self.data_loader.images_labels['train'] = zip(
                *c)

        for img_path, img_label in zip(self.data_loader.images_paths['train'],
                                       self.data_loader.images_labels['train']):
            if only_centers:
                if img_label not in classes_encodings:
                    classes_encodings[img_label] = []
            else:
                if img_label not in classes_counter:
                    classes_counter[img_label] = 0
            if classes_counter[img_label] < max_num_samples_of_each_class:
                encod = self._generate_encoding(img_path)
                
                if encod is not None:
                    if only_centers:
                        classes_encodings[img_label].append(encod)
                    else:
                        data_paths.append(img_path)
                        data_labels.append(img_label)
                        data_encodings.append(encod)
                        classes_counter[img_label] += 1
        if only_centers:
            for class_i, encodings_i in classes_encodings.items():
                encodings_i_np = np.array(encodings_i)
                class_encoding = np.mean(encodings_i_np, axis = 0)
                data_encodings.append(class_encoding)
                data_labels.append(class_i)
        self.encoded_training_data['paths'] = data_paths
        self.encoded_training_data['labels'] = data_labels
        self.encoded_training_data['encodings'] = np.squeeze(
            np.array(data_encodings))
        self.encoded_training_data['knn_classifier'] = KNeighborsClassifier(
            n_neighbors=k_val)
        self.encoded_training_data['knn_classifier'].fit(self.encoded_training_data['encodings'],
                                                         self.encoded_training_data['labels'])
        f = open(save_file_name, "wb")
        pickle.dump(self.encoded_training_data, f)
        f.close()

    def load_encodings(self, path_to_encodings):
        self.encoded_training_data = load_encodings(path_to_encodings)

    def load_model(self, file_path):
        from keras_radam import RAdam
        self.model = load_model(file_path,
                                custom_objects={'contrastive_loss': lac.contrastive_loss,
                                                'accuracy': lac.accuracy,
                                                'loss_function': lac.triplet_loss(self.margin),
                                                'RAdam': RAdam})
        self.input_shape = list(self.model.inputs[0].shape[1:])
        self.base_model = Model(inputs=[self.model.layers[3].get_input_at(0)],
                                outputs=[self.model.layers[3].layers[-1].output])
        self.base_model._make_predict_function()

    def calculate_distances(self, encoding):
        training_encodings = self.encoded_training_data['encodings']
        return np.sqrt(
            np.sum((training_encodings - np.array(encoding))**2, axis=1))

    def predict(self, image):
        if type(image) is str:
            img = cv2.imread(image)
        else:
            img = image
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        distances = self.calculate_distances(encoding)
        max_element = np.argmin(distances)
        predicted_label = self.encoded_training_data['labels'][max_element]
        return predicted_label

    def predict_knn(self, image):
        if type(image) is str:
            img = cv2.imread(image)
        else:
            img = image
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        predicted_label = self.encoded_training_data['knn_classifier'].predict(
            encoding)
        return predicted_label

    def calculate_prediction_accuracy(self):
        correct = 0
        total_n_of_images = len(self.data_loader.images_paths['val'])
        for img_path, img_label in zip(self.data_loader.images_paths['val'],
                                       self.data_loader.images_labels['val']):
            prediction = self.predict_knn(img_path)[0]
            if prediction == img_label:
                correct += 1
        return correct/total_n_of_images
