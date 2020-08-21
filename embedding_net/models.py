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
from .backbones import get_backbone
from . import losses_and_accuracies as lac
from .utils import get_image, get_images
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# TODO
# [] - implement magnet loss
# [] - finalize settings with l1 and l2 losses

class EmbeddingNet:

    def __init__(self,  params):
        self.params_model = params['model']
        self.params_dataloader = params['dataloader']
        self.params_generator = params['generator']
        self.params_general = params['general']
        self.params_train = params['train']
        if 'softmax' in params:
            self.params_softmax = params['softmax']

        self.base_model = None
        self.backbone_model = None
        self.model = None

        self.workdir_path = os.path.join(self.params_general['work_dir'],
                                         self.params_general['project_name'])

        self.encoded_training_data = {}

    def _create_base_model(self):
        self.base_model, self.backbone_model = get_backbone(**self.params_model)
        output = Dense(units=1, activation='sigmoid', name='output_img')(self.base_model.layers[-1].output)
        self.classification_model = Model(inputs=[self.base_model.layers[0].input],outputs=[output])
    
    def _generate_encodings(self, imgs):
        encodings = self.base_model.predict(imgs)
        return encodings
        

    def train_embeddings_classifier(self, data_loader,
                                    classification_model,
                                    max_n_samples=10,
                                    shuffle=True):
        encodings = self.generate_encodings(data_loader, max_n_samples=max_n_samples,  
                                shuffle=shuffle)
        classification_model.fit(encodings['encodings'], 
                                 encodings['labels'])

    def generate_encodings(self, data_loader, max_n_samples=10,  
                                 shuffle=True):
        data_paths, data_labels, data_encodings = [], [], []
        encoded_training_data = {}

        for class_name in data_loader.class_names:
            data_list = data_loader.train_data[class_name]
            if len(data_list)>max_n_samples:
                if shuffle:
                    random.shuffle(data_list)
                data_list = data_list[:max_n_samples]
            
            data_paths += data_list
            imgs = get_images(data_list, self.params_model['input_shape'])
            encods = self._generate_encodings(imgs)
            for encod in encods:
                data_encodings.append(encod)
                data_labels.append(class_name)

        encoded_training_data['paths'] = data_paths
        encoded_training_data['labels'] = data_labels
        encoded_training_data['encodings'] = np.squeeze(np.array(data_encodings))
        
        return encoded_training_data

    def save_encodings(self, encoded_training_data,
                             save_folder='./',
                             save_file_name='encodings.pkl'):
        with open(os.path.join(save_folder, save_file_name), "wb") as f:
            pickle.dump(encoded_training_data, f)

    def load_model(self, file_path):
        import efficientnet.tfkeras as efn
        self.model = load_model(file_path, compile=False)
        model_layers = [x for x in self.model.layers[::-1] if isinstance(x, Model)]
        self.input_shape = list(self.model.inputs[0].shape[1:])
        self.base_model = Model(inputs=[model_layers[0].input],
                                outputs=[model_layers[0].output])
        # self.classification_model = Model(inputs=[self.model.layers[3].get_input_at(0)],
        #                         outputs=[self.model.layers[-1].output])
        # self.classification_model._make_predict_function()
        # self.base_model._make_predict_function()


    def save_base_model(self, save_folder):
        self.base_model.save(f'{save_folder}base_model.h5')

    def save_onnx(self, save_folder, save_name='base_model.onnx'):
        os.environ["TF_KERAS"] = '1'
        import efficientnet.tfkeras as efn
        import keras2onnx
        onnx_model = keras2onnx.convert_keras(self.base_model, self.base_model.name)
        keras2onnx.save_model(onnx_model, os.path.join(save_folder, save_name))

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

    def calculate_prediction_accuracy(self, data_loader):
        correct_top1 = 0
        correct_top5 = 0

        accuracies = {'top1':0,
                      'top5':0 }
        total_n_of_images = len(data_loader.images_paths['val'])
        for img_path, img_label in zip(data_loader.images_paths['val'],
                                       data_loader.images_labels['val']):
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

        self.training = training

        if self.training:
            self._create_base_model()
            self._create_model_triplet()

    
    def _create_model_triplet(self):
        input_image_a = Input(self.params_model['input_shape'])
        input_image_p = Input(self.params_model['input_shape'])
        input_image_n = Input(self.params_model['input_shape'])

        image_encoding_a = self.base_model(input_image_a)
        image_encoding_p = self.base_model(input_image_p)
        image_encoding_n = self.base_model(input_image_n)

        merged_vector = concatenate([image_encoding_a, image_encoding_p, image_encoding_n],axis=-1, name='merged_layer')
        self.model = Model(inputs=[input_image_a, input_image_p, input_image_n],outputs=merged_vector)

        print('Whole model summary')
        self.model.summary()


class SiameseNet(EmbeddingNet):

    def __init__(self, params, training):
        super().__init__(params)
        
        self.training = training

        if self.training:
            self._create_base_model()
            self._create_model_siamese()

    def _create_model_siamese(self):

        input_image_1 = Input(self.params_model['input_shape'])
        input_image_2 = Input(self.params_model['input_shape'])

        image_encoding_1 = self.base_model(input_image_1)
        image_encoding_2 = self.base_model(input_image_2)

        Cl_out1 = Lambda(lambda x: x, name='output_im1')
        Cl_out2 = Lambda(lambda x: x, name='output_im2')

        classification_output_1 = Cl_out1(self.classification_model(input_image_1))
        classification_output_2 = Cl_out2(self.classification_model(input_image_2))

        if self.params_model['distance_type'] == 'l1':
            L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
            distance = L1_layer([image_encoding_1, image_encoding_2])

            embeddings_output = Dense(units=1, activation='sigmoid', name='output_siamese')(distance)

        elif self.params_model['distance_type'] == 'l2':

            L2_layer = Lambda(lambda tensors: K.sqrt(K.maximum(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True), K.epsilon())))
            distance = L2_layer([image_encoding_1, image_encoding_2])

            embeddings_output = distance

        self.model = Model(inputs=[input_image_1, input_image_2], outputs=[embeddings_output, classification_output_1, classification_output_2])

        print('Base model summary')
        self.base_model.summary()

        print('Whole model summary')
        self.model.summary()