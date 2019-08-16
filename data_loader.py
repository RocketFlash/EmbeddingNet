import os
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


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

    def _get_images_set(self, clsss, idxs, s='train', with_aug=True):

        indxs = [self.indexes[s][cl][idx] for cl, idx in zip(clsss, idxs)]
        imgs = [cv2.imread(self.images_paths[s][idx]) for idx in indxs]

        if self.input_shape:
            imgs = [cv2.resize(
                img, (self.input_shape[0], self.input_shape[1])) for img in imgs]

        if with_aug:
            imgs = [self.augmentations(image=img)['image'] for img in imgs]

        return imgs

    def get_batch_pairs(self, batch_size,  s='train'):
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
            imgs = self._get_images_set(
                [selected_class, selected_class], [idx1, idx2], s=s, with_aug=with_aug)
            pairs[0][count, :, :, :] = imgs[0]
            pairs[1][count, :, :, :] = imgs[1]
            targets[i] = 1
            count += 1

        for i in range(n_same_class, batch_size):
            another_class_idx = (
                selected_class_idx + random.randrange(1, self.n_classes)) % self.n_classes
            another_class = self.classes[another_class_idx]
            another_class_n_elements = len(self.indexes[s][another_class])
            idx1 = indxs[i]
            idx2 = random.randrange(0, another_class_n_elements)
            imgs = self._get_images_set(
                [selected_class, another_class], [idx1, idx2], s=s, with_aug=with_aug)
            pairs[0][count, :, :, :] = imgs[0]
            pairs[1][count, :, :, :] = imgs[1]
            targets[i] = 0
            count += 1

        return pairs, targets

    def get_batch_triplets(self, batch_size,  s='train'):
        triplets = [np.zeros((batch_size, self.input_shape[0], self.input_shape[1], 3)),
                    np.zeros(
                        (batch_size, self.input_shape[0], self.input_shape[1], 3)),
                    np.zeros((batch_size, self.input_shape[0], self.input_shape[1], 3))]
        targets = np.zeros((batch_size,))

        count = 0

        for i in range(batch_size):
            selected_class_idx = random.randrange(0, self.n_classes)
            selected_class = self.classes[selected_class_idx]
            selected_class_n_elements = len(self.indexes[s][selected_class])
            another_class_idx = (
                selected_class_idx + random.randrange(1, self.n_classes)) % self.n_classes
            another_class = self.classes[another_class_idx]
            another_class_n_elements = len(self.indexes[s][another_class])

            indxs = np.random.randint(
                selected_class_n_elements, size=batch_size)

            with_aug = s == 'train' and self.augmentations
            idx1 = indxs[i]
            idx2 = (idx1 + random.randrange(1, selected_class_n_elements)
                    ) % selected_class_n_elements
            idx3 = random.randrange(0, another_class_n_elements)
            imgs = self._get_images_set(
                [selected_class, selected_class, another_class], [idx1, idx2, idx3], s=s, with_aug=with_aug)

            triplets[0][count, :, :, :] = imgs[0]
            triplets[1][count, :, :, :] = imgs[1]
            triplets[2][count, :, :, :] = imgs[2]
            targets[i] = 1
            count += 1

        return triplets, targets

    def generate(self, batch_size, mode='siamese', s='train'):
        while True:
            if mode == 'siamese':
                data, targets = self.get_batch_pairs(batch_size, s)
            elif mode == 'triplet':
                data, targets = self.get_batch_triplets(batch_size, s)
            yield (data, targets)

    def get_image(self, img_path):
        img = cv2.imread(img_path)
        if self.input_shape:
            img = cv2.resize(
                img, (self.input_shape[0], self.input_shape[1]))
        return img

    def plot_batch(self, batch_size, mode='triplets', s='train'):
        if mode == 'triplets':
            data, targets = self.get_batch_triplets(batch_size, s)
        else:
            data, targets = self.get_batch_pairs(batch_size, s)
        num_imgs = data[0].shape[0]
        it_val = 3 if mode == 'triplets' else 2
        fig, axs = plt.subplots(num_imgs, it_val, figsize=(
            30, 50), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)

        axs = axs.ravel()
        i = 0
        for img_idx, targ in zip(range(num_imgs), targets):
            for j in range(it_val):
                img = cv2.cvtColor(data[j][img_idx].astype(
                    np.uint8), cv2.COLOR_BGR2RGB)
                axs[i+j].imshow(img)
                axs[i+j].set_title(targ)
            i += it_val

        plt.show()
