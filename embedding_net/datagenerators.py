import os
import cv2
import numpy as np
import random
import pandas as pd
from itertools import combinations
from sklearn.metrics import pairwise_distances
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from .utils import get_image
from tensorflow.keras import backend as K
import tensorflow as tf
import tqdm
import pickle

class ENDataLoader():
    def __init__(self, dataset_path,
                       train_csv_file=None,
                       val_csv_file=None,
                       image_id_column = 'image_id',
                       label_column = 'label',
                       validate = True,
                       val_ratio = 0.1,
                       is_google=False):

        self.dataset_path = dataset_path
        self.class_files_paths = {}
        self.class_names = []
        
        if train_csv_file is not None:
            self.class_files_paths = self._load_from_dataframe(train_csv_file, image_id_column, label_column, is_google)
        else:
            self.class_files_paths = self._load_from_directory()
        
        self.n_classes = len(self.class_names)
        self.n_samples = {k: len(v) for k, v in self.class_files_paths.items()}

        self.validate = validate
        self.val_ratio = val_ratio

        if self.validate:
            if val_csv_file is not None:
                self.train_data = self.class_files_paths
                self.val_data = self._load_from_dataframe(val_csv_file, image_id_column, label_column, is_google)
            else:
                self.train_data, self.val_data = self.split_train_val(self.val_ratio)
        else:
            self.train_data = self.class_files_paths
            self.val_data = {}

    def split_train_val(self, val_ratio):
        train_data = {}
        val_data = {}
        for k, v in self.class_files_paths.items():
            train_d, val_d = train_test_split(v, test_size=val_ratio, random_state=42)
            train_data[k] = train_d
            val_data[k] = val_d
        return train_data, val_data

    def _load_from_dataframe(self, csv_file, image_id_column, label_column, is_google):
        class_files_paths = {}

        # Load data from file if it's already created
        os.makedirs('tmp' , exist_ok=True)
        if os.path.isfile('tmp/data.pickle'):
            print('LOAD DATA FROM FILE')
            with open('tmp/data.pickle', 'rb') as f:
                class_files_paths = pickle.load(f)
            self.class_names = list(class_files_paths.keys())
            print('LOADING DATA FROM FILE COMPLETED')
            return class_files_paths

        dataframe = pd.read_csv(csv_file)
        self.class_names = list(dataframe[label_column].unique())

        for class_name in tqdm.tqdm(self.class_names):
            image_names = dataframe.loc[dataframe[label_column] == class_name][image_id_column]
            if is_google:
                image_paths = [os.path.join(self.dataset_path,f'{f[0]}/{f[1]}/{f[2]}/', f+'.jpg') for f in image_names]
            else:
                image_paths = [os.path.join(self.dataset_path, f) for f in image_names]
            class_files_paths[class_name] = image_paths

        # Save data to file for fast loading
        with open('tmp/data.pickle', 'wb') as f:
            pickle.dump(class_files_paths, f)
        return class_files_paths      

    def _load_from_directory(self):
        class_files_paths = {}
        self.class_names = [f.name for f in os.scandir(self.dataset_path) if f.is_dir()]
        class_dir_paths = [f.path for f in os.scandir(self.dataset_path) if f.is_dir()]

        for class_name, class_dir_path in tqdm.tqdm(zip(self.class_names, class_dir_paths)):
            subdirs = [f.path for f in os.scandir(class_dir_path) if f.is_dir()]
            temp_list = []
            if len(subdirs)>0:
                for subdir in subdirs:
                    class_image_paths = [f.path for f in os.scandir(subdir) if f.is_file() and
                                        (f.name.endswith('.jpg') or
                                        f.name.endswith('.png') and 
                                        not f.name.startswith('._'))]
                    temp_list.extend(class_image_paths)
            else:  
                class_image_paths = [f.path for f in os.scandir(class_dir_path) if f.is_file() and
                                        (f.name.endswith('.jpg') or
                                        f.name.endswith('.png') and 
                                        not f.name.startswith('._'))]
                temp_list.extend(class_image_paths)
            class_files_paths[class_name] = temp_list
        return class_files_paths


class ENDataGenerator(Sequence):
    def __init__(self, class_files_paths,
                       class_names,
                       val_gen = False,
                       input_shape=None,
                       batch_size = 32,
                       n_batches = 10, 
                       n_batches_val = 10,
                       augmentations=None):
        
        self.input_shape = input_shape
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.n_batches_val = n_batches_val
        self.val_gen = val_gen
        self.class_files_paths = class_files_paths
        self.class_names = class_names
        
        self.n_classes = len(self.class_names)
        self.n_samples = {k: len(v) for k, v in self.class_files_paths.items()}

    def __len__(self):
        if self.val_gen:
            return self.n_batches_val
        else:
            return self.n_batches

    def __getitem__(self, index):
        pass       

    def _get_images_set(self, clsss, idxs, with_aug=True):
        if type(clsss) is list:
            img_paths = [self.class_files_paths[cl][idx] for cl, idx in zip(clsss, idxs)]
        else:
            img_paths = [self.class_files_paths[clsss][idx] for idx in idxs]

        imgs = [get_image(img_path, self.input_shape) for img_path in img_paths]

        if with_aug:
            imgs = [self.augmentations(image=img)['image'] for img in imgs]

        return np.array(imgs)/255.


class TripletsDataGenerator(ENDataGenerator):

    def __init__(self, embedding_model,
                       class_files_paths,
                       class_names,
                       n_batches = 10,
                       input_shape=None,
                       batch_size = 32,
                       augmentations=None,
                       k_classes=5,
                       k_samples=5,
                       margin=0.5,
                       negatives_selection_mode='semihard'):
        super().__init__(class_files_paths=class_files_paths, 
                         class_names=class_names, 
                         input_shape=input_shape, 
                         batch_size=batch_size, 
                         n_batches=n_batches, 
                         augmentations=augmentations)
        modes = {'semihard' : self.semihard_negative,
                 'hardest': self.hardest_negative,
                 'random_hard': self.random_hard_negative}
        self.embedding_model = embedding_model

        self.k_classes = k_classes
        self.k_samples = k_samples
        self.margin = margin
        self.negative_selection_fn = modes[negatives_selection_mode]

    def hardest_negative(self, loss_values, margin=0.5):
        hard_negative = np.argmax(loss_values)
        return hard_negative if loss_values[hard_negative] > 0 else None

    def random_hard_negative(self, loss_values, margin=0.5):
        hard_negatives = np.where(loss_values > 0)[0]
        return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

    def semihard_negative(self, loss_values, margin=0.5):
        semihard_negatives = np.where(np.logical_and(
            loss_values < margin, loss_values > 0))[0]
        return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

    def get_batch_triplets_mining(self):
        selected_classes_idxs = np.random.choice(self.n_classes, size=self.k_classes, replace=False)
        selected_classes = [self.class_names[cl] for cl in selected_classes_idxs]
        selected_classes_n_elements = [self.n_samples[cl] for cl in selected_classes]
        selected_images = [np.random.choice(cl_n, size=self.k_samples, replace=True) for cl_n in selected_classes_n_elements]

        all_embeddings_list = []
        all_images_list = []

        
        for idx, cl_img_idxs in enumerate(selected_images):
            images = self._get_images_set(selected_classes[idx], cl_img_idxs, with_aug=self.augmentations)
            all_images_list.append(images)
            embeddings = self.embedding_model.predict(images)
            all_embeddings_list.append(embeddings)

        all_embeddings = np.vstack(all_embeddings_list)
        all_images = np.vstack(all_images_list)
        distance_matrix = pairwise_distances(all_embeddings)

        triplet_anchors = []
        triplet_positives = []
        triplet_negatives = []
        targets = []
        for idx, _ in enumerate(selected_classes):
            current_class_mask = np.zeros(self.k_classes*self.k_samples, dtype=bool)
            current_class_mask[idx*self.k_samples:(idx+1)*self.k_samples] = True
            other_classes_mask = np.logical_not(current_class_mask)
            positive_indices = np.where(current_class_mask)[0]
            negative_indices = np.where(other_classes_mask)[0]
            anchor_positives = np.array(list(combinations(positive_indices, 2)))

            ap_distances = distance_matrix[anchor_positives[:,0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[anchor_positive[0], negative_indices] + self.margin
                loss_values = np.array(loss_values)
                hard_negative = self.negative_selection_fn(loss_values, margin=self.margin)

                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplet_anchors.append(all_images[anchor_positive[0]])
                    triplet_positives.append(all_images[anchor_positive[1]])
                    triplet_negatives.append(all_images[hard_negative])
                    targets.append(1)

        if len(triplet_anchors) == 0:
            triplet_anchors.append(all_images[anchor_positive[0]])
            triplet_positives.append(all_images[anchor_positive[1]])
            triplet_negatives.append(all_images[negative_indices[0]])
            targets.append(1)

        triplet_anchors = np.array(triplet_anchors)
        triplet_positives = np.array(triplet_positives)
        triplet_negatives = np.array(triplet_negatives)
        targets = np.array(targets)

        triplets = [triplet_anchors, triplet_positives, triplet_negatives]
        return triplets, targets

    def __getitem__(self, index):
        return self.get_batch_triplets_mining()


class SimpleTripletsDataGenerator(ENDataGenerator):
    def __init__(self, class_files_paths,
                       class_names,
                       input_shape=None,
                       batch_size = 32,
                       n_batches = 10, 
                       augmentations=None,
                       **kwargs):
        super().__init__(class_files_paths=class_files_paths, 
                         class_names=class_names, 
                         input_shape=input_shape, 
                         batch_size=batch_size, 
                         n_batches=n_batches, 
                         augmentations=augmentations)

    def get_batch_triplets(self):
        triplets = [np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3)),
                    np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3)),
                    np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3))]
        targets = np.zeros((self.batch_size,))

        count = 0

        for i in range(self.batch_size):
            selected_class_idx = random.randrange(0, self.n_classes)
            selected_class = self.class_names[selected_class_idx]
            selected_class_n_elements = self.n_samples[selected_class]
            another_class_idx = (
                selected_class_idx + random.randrange(1, self.n_classes)) % self.n_classes
            another_class = self.class_names[another_class_idx]
            another_class_n_elements = self.n_samples[another_class]

            idx1 = random.randrange(0, selected_class_n_elements)
            idx2 = (idx1 + random.randrange(1, selected_class_n_elements)
                    ) % selected_class_n_elements
            idx3 = random.randrange(0, another_class_n_elements)

            imgs = self._get_images_set([selected_class, selected_class, another_class], 
                                        [idx1, idx2, idx3], 
                                        with_aug=self.augmentations)

            triplets[0][count, :, :, :] = imgs[0]
            triplets[1][count, :, :, :] = imgs[1]
            triplets[2][count, :, :, :] = imgs[2]
            targets[i] = 1
            count += 1

        return triplets, targets

    def __getitem__(self, index):
        return self.get_batch_triplets()


class SiameseDataGenerator(ENDataGenerator):

    def __init__(self, class_files_paths,
                       class_names,
                       val_gen = False,
                       input_shape=None,
                       batch_size = 32,
                       n_batches = 10,
                       n_batches_val = 10, 
                       augmentations=None):

        super().__init__(class_files_paths=class_files_paths, 
                         class_names=class_names,
                         val_gen = False, 
                         input_shape=input_shape, 
                         batch_size=batch_size, 
                         n_batches=n_batches,
                         n_batches_val = 10, 
                         augmentations=augmentations)

    def get_batch_pairs(self):
        pairs = [np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3)), np.zeros(
            (self.batch_size, self.input_shape[0], self.input_shape[1], 3))]
        targets = np.zeros((self.batch_size, ))
        targets1 = np.zeros((self.batch_size, ))
        targets2 = np.zeros((self.batch_size, ))
        # targets = []

        n_same_class = self.batch_size // 2

        selected_class_idx = random.randrange(0, self.n_classes)
        selected_class = self.class_names[selected_class_idx]
        selected_class_n_elements = self.n_samples[selected_class] 

        indxs = np.random.randint(selected_class_n_elements, size=self.batch_size)

        with_aug = self.augmentations
        count = 0
        for i in range(n_same_class):
            idx1 = indxs[i]
            idx2 = (idx1 + random.randrange(1, selected_class_n_elements)) % selected_class_n_elements
            imgs = self._get_images_set([selected_class, selected_class], [idx1, idx2], with_aug=with_aug)
            pairs[0][count, :, :, :] = imgs[0]
            pairs[1][count, :, :, :] = imgs[1]
            targets[i] = 1
            count += 1

        for i in range(n_same_class, self.batch_size):
            another_class_idx = (selected_class_idx + random.randrange(1, self.n_classes)) % self.n_classes
            another_class = self.class_names[another_class_idx]
            another_class_n_elements = self.n_samples[another_class]
            idx1 = indxs[i]
            idx2 = random.randrange(0, another_class_n_elements)
            imgs = self._get_images_set([selected_class, another_class], [idx1, idx2], with_aug=with_aug)
            pairs[0][count, :, :, :] = imgs[0]
            pairs[1][count, :, :, :] = imgs[1]
            targets[i] = 0
            count += 1
        return pairs, targets

    def __getitem__(self, index):
        return self.get_batch_pairs()


class SimpleDataGenerator(ENDataGenerator):
    def __init__(self, class_files_paths,
                       class_names,
                       input_shape=None,
                       batch_size = 32,
                       n_batches = 10, 
                       augmentations=None):

        super().__init__(class_files_paths=class_files_paths, 
                         class_names=class_names, 
                         input_shape=input_shape, 
                         batch_size=batch_size, 
                         n_batches=n_batches, 
                         augmentations=augmentations)
    
    def get_batch(self):
        images = [
            np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3))]
        targets = np.zeros((self.batch_size, self.n_classes))

        count = 0
        with_aug = self.augmentations
        for i in range(self.batch_size):
            selected_class_idx = random.randrange(0, self.n_classes)
            selected_class = self.class_names[selected_class_idx]
            selected_class_n_elements = len(self.class_files_paths[selected_class])

            indx = random.randrange(0, selected_class_n_elements)

            img = self._get_images_set([selected_class], [indx], with_aug=with_aug)
            images[0][count, :, :, :] = img[0]
            targets[i][selected_class_idx] = 1
            count += 1

        return images, targets

    def __getitem__(self, index):
        return self.get_batch()
