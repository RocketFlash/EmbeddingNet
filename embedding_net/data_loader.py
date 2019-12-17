import os
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from itertools import combinations
from sklearn.metrics import pairwise_distances


class EmbeddingNetImageLoader:
    """
    Image loader for Embedding network
    """

    def __init__(self, dataset_path, input_shape=None, augmentations=None, data_subsets=['train', 'val']):
        self.dataset_path = dataset_path
        self.data_subsets = data_subsets
        self.images_paths = {}
        self.images_labels = {}
        self.input_shape = input_shape
        self.augmentations = augmentations
        self.current_idx = {d: 0 for d in data_subsets}
        self._load_images_paths()
        self.classes = {
            s: sorted(list(set(self.images_labels[s]))) for s in data_subsets}
        self.n_classes = {s: len(self.classes[s]) for s in data_subsets}
        self.n_samples = {d: len(self.images_paths[d]) for d in data_subsets}
        self.indexes = {d: {cl: np.where(np.array(self.images_labels[d]) == cl)[
            0] for cl in self.classes[d]} for d in data_subsets}

    def _load_images_paths(self):
        for d in self.data_subsets:
            self.images_paths[d] = []
            self.images_labels[d] = []
            for root, dirs, files in os.walk(self.dataset_path+d):
                for f in files:
                    if f.endswith('.jpg') or f.endswith('.png') and not f.startswith('._'):
                        self.images_paths[d].append(root+'/'+f)
                        self.images_labels[d].append(root.split('/')[-1])

    def _get_images_set(self, clsss, idxs, s='train', with_aug=True):
        if type(clsss) is list:
            indxs = [self.indexes[s][cl][idx] for cl, idx in zip(clsss, idxs)]
        else:
            indxs = [self.indexes[s][clsss][idx] for idx in idxs]
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

        selected_class_idx = random.randrange(0, self.n_classes[s])
        selected_class = self.classes[s][selected_class_idx]
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
                selected_class_idx + random.randrange(1, self.n_classes[s])) % self.n_classes[s]
            another_class = self.classes[s][another_class_idx]
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
            selected_class_idx = random.randrange(0, self.n_classes[s])
            selected_class = self.classes[s][selected_class_idx]
            selected_class_n_elements = len(self.indexes[s][selected_class])
            another_class_idx = (
                selected_class_idx + random.randrange(1, self.n_classes[s])) % self.n_classes[s]
            another_class = self.classes[s][another_class_idx]
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

    def get_batch_triplets_batch_all(self):
        pass

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

    def get_batch_triplets_mining(self,
                                  embedding_model,
                                  n_classes,
                                  n_samples,
                                  margin=0.5,
                                  negative_selection_mode='semihard',
                                  s='train'):
        if negative_selection_mode == 'semihard':
            negative_selection_fn = self.semihard_negative
        elif negative_selection_mode == 'hardest':
            negative_selection_fn = self.hardest_negative
        else:
            negative_selection_fn = self.random_hard_negative

        selected_classes_idxs = np.random.choice(
            self.n_classes[s], size=n_classes, replace=False)
        selected_classes = [self.classes[s][cl]
                            for cl in selected_classes_idxs]
        selected_classes_n_elements = [
            self.indexes[s][cl].shape[0] for cl in selected_classes]

        selected_images = [np.random.choice(
            cl, size=n_samples, replace=False) for cl in selected_classes_n_elements]

        all_embeddings_list = []
        all_images_list = []
        with_aug = s == 'train' and self.augmentations
        for idx, cl_img_idxs in enumerate(selected_images):
            images = self._get_images_set(
                selected_classes[idx], cl_img_idxs, s='train', with_aug=with_aug)
            images = np.array(images)
            all_images_list.append(images)
            embeddings = embedding_model.predict(images)
            all_embeddings_list.append(embeddings)
        all_embeddings = np.vstack(all_embeddings_list)
        all_images = np.vstack(all_images_list)
        distance_matrix = pairwise_distances(all_embeddings)

        triplet_anchors = []
        triplet_positives = []
        triplet_negatives = []
        targets = []
        for idx, selected_class in enumerate(selected_classes):
            current_class_mask = np.zeros(n_classes*n_samples, dtype=bool)
            current_class_mask[idx*n_samples:(idx+1)*n_samples] = True
            other_classes_mask = np.logical_not(current_class_mask)
            positive_indices = np.where(current_class_mask)[0]
            negative_indices = np.where(other_classes_mask)[0]
            anchor_positives = np.array(
                list(combinations(positive_indices, 2)))

            ap_distances = distance_matrix[anchor_positives[:,
                                                            0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - \
                    distance_matrix[anchor_positive[0],
                                    negative_indices] + margin
                loss_values = np.array(loss_values)
                hard_negative = negative_selection_fn(
                    loss_values, margin=margin)
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

    def generate(self, batch_size, mode='siamese', s='train'):
        while True:
            if mode == 'siamese':
                data, targets = self.get_batch_pairs(batch_size, s)
            elif mode == 'triplet':
                data, targets = self.get_batch_triplets(batch_size, s)
            yield (data, targets)

    def generate_mining(self, embedding_model, n_classes, n_samples, margin=0.5, negative_selection_mode='semihard', s='train'):
        while True:
            data, targets = self.get_batch_triplets_mining(embedding_model,
                                                           n_classes,
                                                           n_samples,
                                                           margin=margin,
                                                           negative_selection_mode='semihard',
                                                           s=s)
            yield (data, targets)

    def get_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print('image is not exist ' + img_path)
            return None
        if self.input_shape:
            img = cv2.resize(
                img, (self.input_shape[0], self.input_shape[1]))
        return img

    def plot_batch(self, data, targets):
        num_imgs = data[0].shape[0]
        it_val = len(data)
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


class SimpleNetImageLoader:
    """
    Image loader for Embedding network
    """

    def __init__(self, dataset_path, input_shape=None, augmentations=None, data_subsets=['train', 'val']):
        self.dataset_path = dataset_path
        self.data_subsets = data_subsets
        self.images_paths = {}
        self.images_labels = {}
        self.input_shape = input_shape
        self.augmentations = augmentations
        self.current_idx = {d: 0 for d in data_subsets}
        self._load_images_paths()
        self.classes = {
            s: sorted(list(set(self.images_labels[s]))) for s in data_subsets}
        self.n_classes = {s: len(self.classes[s]) for s in data_subsets}
        self.n_samples = {d: len(self.images_paths[d]) for d in data_subsets}
        self.indexes = {d: {cl: np.where(np.array(self.images_labels[d]) == cl)[
            0] for cl in self.classes[d]} for d in data_subsets}

    def _load_images_paths(self):
        for d in self.data_subsets:
            self.images_paths[d] = []
            self.images_labels[d] = []
            for root, dirs, files in os.walk(self.dataset_path+d):
                for f in files:
                    if f.endswith('.jpg') or f.endswith('.png') and not f.startswith('._'):
                        self.images_paths[d].append(root+'/'+f)
                        self.images_labels[d].append(root.split('/')[-1])

    def _get_images_set(self, clsss, idxs, s='train', with_aug=True):
        if type(clsss) is list:
            indxs = [self.indexes[s][cl][idx] for cl, idx in zip(clsss, idxs)]
        else:
            indxs = [self.indexes[s][clsss][idx] for idx in idxs]

        imgs = [cv2.imread(self.images_paths[s][idx]) for idx in indxs]

        if self.input_shape:
            imgs = [cv2.resize(
                img, (self.input_shape[0], self.input_shape[1])) for img in imgs]

        if with_aug:
            imgs = [self.augmentations(image=img)['image'] for img in imgs]

        return imgs

    def get_batch(self, batch_size,  s='train'):
        images = [
            np.zeros((batch_size, self.input_shape[0], self.input_shape[1], 3))]
        targets = np.zeros((batch_size, self.n_classes[s]))

        count = 0
        with_aug = s == 'train' and self.augmentations
        for i in range(batch_size):
            selected_class_idx = random.randrange(0, self.n_classes[s])
            selected_class = self.classes[s][selected_class_idx]
            selected_class_n_elements = len(self.indexes[s][selected_class])

            indx = random.randrange(0, selected_class_n_elements)

            img = self._get_images_set(
                [selected_class], [indx], s=s, with_aug=with_aug)
            images[0][count, :, :, :] = img[0]
            targets[i][selected_class_idx] = 1
            count += 1

        return images, targets

    def generate(self, batch_size, s='train'):
        while True:
            data, targets = self.get_batch(batch_size, s)
            yield (data, targets)

    def get_image(self, img_path):
        img = cv2.imread(img_path)
        if self.input_shape:
            img = cv2.resize(
                img, (self.input_shape[0], self.input_shape[1]))
        return img

    def plot_batch(self, data, targets):
        num_imgs = data[0].shape[0]
        it_val = len(data)
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
        num_imgs = data[0].shape[0]
        it_val = len(data)
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
