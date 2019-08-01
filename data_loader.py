import os
import cv2
import numpy as np
import random


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
        self.classes = list(set(self.images_labels['train']))
        self.n_classes = len(self.classes)
        self.n_samples = {d: len(self.images_paths[d]) for d in data_subsets}
        self.indexes = {d: {cl: np.where(np.array(self.images_labels[d]) == cl)[
            0] for cl in self.classes} for d in data_subsets}

    def _load_images_paths(self):
        for d in self.data_subsets:
            self.images_paths[d] = []
            self.images_labels[d] = []
            for root, dirs, files in os.walk(self.dataset_path+d):
                for f in files:
                    if f.endswith('.jpg') or f.endswith('.png'):
                        self.images_paths[d].append(root+'/'+f)
                        self.images_labels[d].append(root.split('/')[-1])

    def _get_pair(self, cl1, cl2, idx1, idx2, s='train', with_aug=True):
        indx_1 = self.indexes[s][cl1][idx1]
        indx_2 = self.indexes[s][cl2][idx2]
        img_1 = cv2.imread(self.images_paths[s][indx_1])
        img_2 = cv2.imread(self.images_paths[s][indx_2])
        if self.input_shape:
            img_1 = cv2.resize(
                img_1, (self.input_shape[0], self.input_shape[1]))
            img_2 = cv2.resize(
                img_2, (self.input_shape[0], self.input_shape[1]))
        if with_aug:
            img_1 = self.augmentations(image=img_1)['image']
            img_2 = self.augmentations(image=img_2)['image']
        return img_1, img_2

    def get_batch(self, batch_size, s='train'):
        pairs = [np.zeros((batch_size, self.input_shape[0], self.input_shape[1], 3)), np.zeros(
            (batch_size, self.input_shape[0], self.input_shape[1], 3))]
        targets = np.zeros((batch_size,))

        n_same_class = batch_size // 2

        selected_class_idx = random.randrange(0, self.n_classes)
        selected_class = self.classes[selected_class_idx]
        selected_class_n_elements = len(self.indexes[s][selected_class])

        indxs = np.random.randint(
            selected_class_n_elements, size=batch_size)

        with_aug = s == 'train' and self.augmentations
        count = 0
        for i in range(n_same_class):
            idx1 = indxs[i]
            idx2 = (idx1 + random.randrange(1, selected_class_n_elements)
                    ) % selected_class_n_elements
            img1, img2 = self._get_pair(
                selected_class, selected_class, idx1, idx2, s=s, with_aug=with_aug)
            pairs[0][count, :, :, :] = img1
            pairs[1][count, :, :, :] = img2
            targets[i] = 1
            count += 1

        for i in range(n_same_class, batch_size):
            another_class_idx = (
                selected_class_idx + random.randrange(1, self.n_classes)) % self.n_classes
            another_class = self.classes[another_class_idx]
            another_class_n_elements = len(self.indexes[s][another_class])
            idx1 = indxs[i]
            idx2 = random.randrange(0, another_class_n_elements)
            img1, img2 = self._get_pair(
                selected_class, another_class, idx1, idx2, s=s, with_aug=with_aug)
            pairs[0][count, :, :, :] = img1
            pairs[1][count, :, :, :] = img2
            targets[i] = 0
            count += 1

        return pairs, targets

    def generate(self, batch_size, s="train"):
        while True:
            pairs, targets = self.get_batch(batch_size, s)
            yield (pairs, targets)

    def get_image(self, img_path):
        img = cv2.imread(img_path)
        if self.input_shape:
            img = cv2.resize(
                img, (self.input_shape[0], self.input_shape[1]))
        return img
