import os
import cv2
import numpy as np


class SiameseImageLoader:
    """
    Image loader for Siamese network
    """

    def __init__(self, dataset_path, number_of_classes=2, input_shape=None, augmentations=None, data_subsets=['train', 'val']):
        self.dataset_path = dataset_path
        self.data_subsets = data_subsets
        self.images_paths = {}
        self.images_labels = {}
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes
        self.augmentations = augmentations
        self.current_idx = {d: 0 for d in data_subsets}
        self._load_images_paths()
        self.n_samples = {d: len(self.images_paths[d]) for d in data_subsets}
        self.indexes = {d: np.arange(self.n_samples[d]) for d in data_subsets}
        for d in self.data_subsets:
            np.random.shuffle(self.indexes[d])

    def _load_images_paths(self):
        for d in self.data_subsets:
            self.images_paths[d] = []
            self.images_labels[d] = []
            for root, dirs, files in os.walk(self.dataset_path+d):
                for f in files:
                    if f.endswith('.jpg') or f.endswith('.png'):
                        self.images_paths[d].append(root+'/'+f)
                        self.images_labels[d].append(root.split('/')[-1])

    def get_batch(self, batch_size, s="train"):
        if self.current_idx[s] + 2*batch_size >= self.n_samples[s]:
            np.random.shuffle(self.indexes[s])
            self.current_idx[s] = 0

        pairs_1 = []
        pairs_2 = []

        targets = np.zeros((batch_size,))
        for i in range(batch_size):
            indx_1 = self.indexes[s][self.current_idx[s] + 2 * i]
            indx_2 = self.indexes[s][self.current_idx[s] + 2 * i + 1]
            img_1 = cv2.imread(
                self.images_paths[s][indx_1])
            img_2 = cv2.imread(
                self.images_paths[s][indx_2])
            if self.input_shape:
                img_1 = cv2.resize(
                    img_1, (self.input_shape[0], self.input_shape[1]))
                img_2 = cv2.resize(
                    img_2, (self.input_shape[0], self.input_shape[1]))
            if s == 'train' and self.augmentations:
                img_1 = self.augmentations(image=img_1)['image']
                img_2 = self.augmentations(image=img_2)['image']

            pairs_1.append(img_1)
            pairs_2.append(img_2)

            if self.images_labels[s][indx_1] == self.images_labels[s][indx_2]:
                targets[i] = 1
            else:
                targets[i] = 0
        self.current_idx[s] += 2*batch_size
        pairs = [np.array(pairs_1), np.array(pairs_2)]
        return pairs, targets

    def generate(self, batch_size, s="train"):
        while True:
            pairs, targets = self.get_batch(batch_size, s)
            yield (pairs, targets)
