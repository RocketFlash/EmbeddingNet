import os
import numpy as np
import tensorflow.keras.backend as K
import cv2
import random
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Input, Lambda, concatenate, GlobalAveragePooling2D
import pickle
from .utils import load_encodings, parse_params
from .datagenerators import ENDataLoader, SimpleDataGenerator, TripletsDataGenerator, SimpleTripletsDataGenerator, SiameseDataGenerator
from .backbones import get_backbone
from . import losses_and_accuracies as lac
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class EmbeddingNet:

    def __init__(self,  params):
        self.params_model = params['model']
        self.params_dataloader = params['dataloader']
        self.params_generator = params['generator']
        self.params_save_paths = params['save_paths']
        self.params_train = params['train']
        if 'softmax' in params:
            self.params_softmax = params['softmax']

        self.base_model = None
        self.backbone_model = None

        self.encoded_training_data = {}
        self.dataloader = None
        self.train_generator = None
        self.val_generator = None

    def pretrain_backbone_softmax(self):

        optimizer = self.params_softmax['optimizer']
        learning_rate = self.params_softmax['learning_rate']
        decay_factor = self.params_softmax['decay_factor']
        step_size = self.params_softmax['step_size']

        input_shape = self.params_softmax['input_shape']
        batch_size = self.params_softmax['batch_size']
        val_steps = self.params_softmax['val_steps']
        steps_per_epoch = self.params_softmax['steps_per_epoch']
        n_epochs = self.params_softmax['n_epochs']
        augmentations = self.params_softmax['augmentations']

        n_classes = self.data_loader.n_classes

        x = GlobalAveragePooling2D()(self.backbone_model.output)

        output = Dense(n_classes, activation='softmax')(x)
        model = Model(inputs=[self.backbone_model.input], outputs=[output])

        # train
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        train_generator = SimpleDataGenerator(self.data_loader.train_data,
                                              self.data_loader.class_names,
                                              input_shape=input_shape,
                                              batch_size = batch_size,
                                              n_batches = steps_per_epoch, 
                                              augmentations=augmentations)

        if self.data_loader.validate:
            val_generator = SimpleDataGenerator(self.data_loader.val_data,
                                              self.data_loader.class_names,
                                              input_shape=input_shape,
                                              batch_size = batch_size,
                                              n_batches = steps_per_epoch, 
                                              augmentations=augmentations)
            checkpoint_callback_monitor = 'val_loss'
        else:
            val_generator = None
            checkpoint_callback_monitor = 'loss'

        tensorboard_save_path = os.path.join(
            self.params_save_paths['work_dir'],
            self.params_save_paths['project_name'], 
            'pretraining_model/tf_log/')
        weights_save_file = os.path.join(
            self.params_save_paths['work_dir'],
            self.params_save_paths['project_name'], 
            'pretraining_model/weights/',
            self.params_save_paths['project_name']+'.h5')

        callbacks = [
            LearningRateScheduler(lambda x: learning_rate *
                                decay_factor ** np.floor(x/step_size)),
            ReduceLROnPlateau(monitor=checkpoint_callback_monitor, factor=0.1,
                            patience=20, verbose=1),
            EarlyStopping(monitor=checkpoint_callback_monitor,
                          patience=10, 
                          verbose=1, 
                          restore_best_weights=True),
            TensorBoard(log_dir=tensorboard_save_path),
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

    def _create_base_model(self):
        self.base_model, self.backbone_model = get_backbone(**self.params_model)

    def _create_dataloader(self):
        self.data_loader = ENDataLoader(**self.params_dataloader)

    def _create_generators(self):
        pass
    
    def _generate_encoding(self, img_path):
        img = self.data_loader.get_image(img_path)
        if img is None:
            return None
        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        return encoding
        
    def train(self, callbacks=[], verbose=1):
        history = self.model.fit_generator(self.train_generator,
                                           validation_data=self.val_generator,  
                                           epochs=self.params_train['n_epochs'], 
                                           callbacks=callbacks,
                                           verbose=verbose,
                                           use_multiprocessing=True)

        return history

    def train_embeddings_classifier(self, 
                                    classification_model,
                                    max_n_samples=10,
                                    shuffle=True):
        encodings = self.generate_encodings(max_n_samples=max_n_samples,  
                                shuffle=shuffle)
        classification_model.fit(encodings['encodings'], 
                                 encodings['labels'])

    def generate_encodings(self, max_n_samples=10,  
                                 shuffle=True):
        data_paths, data_labels, data_encodings = [], [], []
        encoded_training_data = {}

        for class_name in self.data_loader.class_names:
            data_list = self.data_loader.train_data[class_name]
            if len(data_list)>max_n_samples:
                if shuffle:
                    random.shuffle(data_list)
                data_list = data_list[:max_n_samples]
            
            data_paths += data_list
            data_labels += class_name * len(data_list)
            for img_path in data_list:
                encod = self._generate_encoding(img_path)
                data_encodings.append(encod)

        encoded_training_data['paths'] = data_paths
        encoded_training_data['labels'] = data_labels
        encoded_training_data['encodings'] = np.squeeze(np.array(data_encodings))
        
        return encoded_training_data

    def save_encodings(self, encoded_training_data,
                             save_file_name='encodings.pkl'):
        with open(save_file_name, "wb") as f:
            pickle.dump(encoded_training_data, f)

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

    def predict(self, image):
        if type(image) is str:
            img = cv2.imread(image)
        else:
            img = image
        img = cv2.resize(img, (self.params_model['input_shape'][0], 
                               self.params_model['input_shape'][1]))
        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        distances = self.calculate_distances(encoding)
        max_element = np.argmin(distances)
        predicted_label = self.encoded_training_data['labels'][max_element]
        return predicted_label

    def predict_knn(self, image, with_top5=False):
        if type(image) is str:
            img = cv2.imread(image)
        else:
            img = image
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))

        encoding = self.base_model.predict(np.expand_dims(img, axis=0))
        predicted_label = self.encoded_training_data['knn_classifier'].predict(encoding)
        if with_top5:    
            prediction_top5_idx = self.encoded_training_data['knn_classifier'].kneighbors(encoding, n_neighbors=5)
            prediction_top5 = [self.encoded_training_data['labels'][prediction_top5_idx[1][0][i]] for i in range(5)]
            return predicted_label, prediction_top5
        else:
            return predicted_label

    def calculate_prediction_accuracy(self):
        correct_top1 = 0
        correct_top5 = 0

        accuracies = {'top1':0,
                      'top5':0 }
        total_n_of_images = len(self.data_loader.images_paths['val'])
        for img_path, img_label in zip(self.data_loader.images_paths['val'],
                                       self.data_loader.images_labels['val']):
            prediction, prediction_top5 = self.predict_knn(img_path, with_top5=True)
            if prediction[0] == img_label:
                correct_top1 += 1
            if img_label in prediction_top5:
                correct_top5 += 1
        accuracies['top1'] = correct_top1/total_n_of_images
        accuracies['top5'] = correct_top5/total_n_of_images

        return accuracies


class TripletNet(EmbeddingNet):

    def __init__(self, params, training=False):
        super().__init__(params)
        self._create_base_model()
        self.base_model._make_predict_function()

        self.training = training

        if self.training:
            self._create_dataloader()
            self._create_generators()
            self._create_model_triplet()

    def _create_generators(self):
        self.train_generator = TripletsDataGenerator(embedding_model=self.base_model,
                                               class_files_paths=self.data_loader.train_data,
                                               class_names=self.data_loader.class_names,
                                               **self.params_generator)
        if self.data_loader.validate:
            self.val_generator = SimpleTripletsDataGenerator(self.data_loader.val_data,
                                               self.data_loader.class_names,
                                               **self.params_generator)
    
    def _create_model_triplet(self):
        input_image_a = Input(self.params_model['input_shape'])
        input_image_p = Input(self.params_model['input_shape'])
        input_image_n = Input(self.params_model['input_shape'])

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

        self.model.compile(loss=lac.triplet_loss(self.params_generator['margin']), 
                                                 optimizer=self.params_train['optimizer'])


class SiameseNet(EmbeddingNet):

    def __init__(self, params, training):
        super().__init__(params)
        self._create_model_siamese()
        if training:
            self.dataloader = {}
            self.train_generator = {}
            self.val_generator = {}
            self._create_generators()

    def _create_generators(self):
        self.train_generator = SiameseDataGenerator(class_files_paths=self.data_loader.train_data,
                                               class_names=self.data_loader.class_names,
                                               **self.params_generator)
        if self.data_loader.validate:
            self.val_generator = SiameseDataGenerator(class_files_paths=self.data_loader.val_data,
                                               class_names=self.data_loader.class_names,
                                               **self.params_generator)

    def _create_model_siamese(self):

        input_image_1 = Input(self.params_model['input_shape'])
        input_image_2 = Input(self.params_model['input_shape'])

        image_encoding_1 = self.base_model(input_image_1)
        image_encoding_2 = self.base_model(input_image_2)

        if self.params_model['distance_type'] == 'l1':
            L1_layer = Lambda(
                lambda tensors: K.abs(tensors[0] - tensors[1]))
            distance = L1_layer([image_encoding_1, image_encoding_2])

            prediction = Dense(units=1, activation='sigmoid')(distance)
            metric = 'binary_accuracy'

        elif self.params_model['distance_type'] == 'l2':

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
                           optimizer=self.params_train['optimizer'])