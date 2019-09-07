import os
import numpy as np
import keras.backend as K
import cv2
import random
from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Dense, Input, Lambda, concatenate
import pickle
from .utils import parse_net_params, load_encodings
from .backbones import get_backbone
from . import losses_and_accuracies as lac
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class EmbeddingNet:
    """
    SiameseNet for image classification
    distance_type = 'l1' -> l1_loss
    distance_type = 'l2' -> l2_loss
    
    mode = 'siamese' -> Siamese network
    mode = 'triplet' -> Triplen network
    """

    def __init__(self,  cfg_file=None):
        if cfg_file:
            params = parse_net_params(cfg_file)
            self.input_shape = params['input_shape']
            self.encodings_len = params['encodings_len']
            self.backbone = params['backbone']
            self.backbone_weights = params['backbone_weights']
            self.distance_type = params['distance_type']
            self.mode = params['mode']
            self.project_name = params['project_name']
            self.optimizer = params['optimizer']
            self.freeze_backbone = params['freeze_backbone']
            self.data_loader = params['loader']
            self.embeddings_normalization = params['embeddings_normalization']
            self.margin = params['margin']

            self.model = []
            self.base_model = []
            self.l_model = []

            self.encodings_path = params['encodings_path']
            self.plots_path = params['plots_path']
            self.tensorboard_log_path = params['tensorboard_log_path']
            self.weights_save_path = params['weights_save_path']
            self.model_save_name = params['model_save_name']

            os.makedirs(self.encodings_path, exist_ok=True)
            os.makedirs(self.plots_path, exist_ok=True)
            os.makedirs(self.tensorboard_log_path, exist_ok=True)
            os.makedirs(self.weights_save_path, exist_ok=True)

            if self.mode == 'siamese':
                self._create_model_siamese()
            elif self.mode == 'triplet':
                self._create_model_triplet()
            
            self.encoded_training_data = {}
        else:
            self.margin = 0.5


    def _create_base_model(self):      
        self.base_model = get_backbone(input_shape=self.input_shape,
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
        self.model  = Model(inputs=[input_image_a,input_image_p, input_image_n], 
                            outputs=merged_vector)
        
        print('Base model summary')
        self.base_model.summary()

        print('Whole model summary')
        self.model.summary()

        self.model.compile(loss=lac.triplet_loss(self.margin), optimizer=self.optimizer)


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

        train_generator = self.data_loader.generate(batch_size, mode=self.mode, s="train")
        val_generator = self.data_loader.generate(batch_size, mode=self.mode, s="val")
        
        history = self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                 verbose=verbose, validation_data = val_generator, validation_steps = val_steps, callbacks=callbacks)
        if self.plots_path:
            self.plot_grapths(history)
        return history
    
    def train_generator_mining(self, 
                               steps_per_epoch, 
                               epochs, callbacks = [], 
                               val_steps=100, 
                               with_val=True, 
                               n_classes=4, 
                               n_samples=4,
                               val_batch=8,
                               negative_selection_mode='semihard', 
                               verbose=1):

        train_generator = self.data_loader.generate_mining(self.base_model, n_classes, n_samples, margin=self.margin, negative_selection_mode=negative_selection_mode, s="train")
        # val_generator = self.data_loader.generate_mining(self.base_model, n_classes, n_samples, negative_selection_mode=negative_selection_mode, s="val")
        val_generator = self.data_loader.generate(val_batch, mode=self.mode, s="val")

        history = self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                 verbose=verbose, validation_data = val_generator, validation_steps = val_steps, callbacks=callbacks)
        if self.plots_path:
            self.plot_grapths(history)
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
        self.encoded_training_data['knn_classifier'] = KNeighborsClassifier(n_neighbors=1)
        self.encoded_training_data['knn_classifier'].fit(self.encoded_training_data['encodings'],
                                                         self.encoded_training_data['labels'])
        f = open(os.path.join(self.encodings_path, save_file_name), "wb")
        pickle.dump(self.encoded_training_data, f)
        f.close()

    def load_encodings(self, path_to_encodings):
        self.encoded_training_data = load_encodings(path_to_encodings)

    def load_model(self,file_path):
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
        predicted_label = self.encoded_training_data['knn_classifier'].predict(encoding)
        return predicted_label

    def calculate_prediction_accuracy(self):
        correct = 0
        total_n_of_images = len(self.data_loader.images_paths['val'])
        for img_path, img_label in zip(self.data_loader.images_paths['val'],
                                       self.data_loader.images_labels['val']):
            prediction = self.predict_knn(img_path)[0]
            if prediction == img_label:
                correct+=1
        return correct/total_n_of_images

    def plot_grapths(self, history):
        for k, v in history.history.items():
            t = list(range(len(v)))
            fig, ax = plt.subplots()
            ax.plot(t, v)

            ax.set(xlabel='epoch', ylabel='{}'.format(k),
                title='{}'.format(k))
            ax.grid()

            fig.savefig("{}{}.png".format(self.plots_path, k))

