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

    def __init__(self, dataset_path, 
                       input_shape=None, 
                       augmentations=None, 
                       min_n_obj_per_class = None,
                       select_max_n_obj_per_class = None, 
                       max_n_obj_per_class = None):
        self.dataset_path = dataset_path
        self.data_subsets = [d.split('/')[-1] for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        self.images_paths = {}
        self.images_labels = {}
        self.input_shape = input_shape
        self.augmentations = augmentations
        self.min_n_obj_per_class = min_n_obj_per_class if min_n_obj_per_class else 0
        self.select_max_n_obj_per_class = select_max_n_obj_per_class if select_max_n_obj_per_class else 1e10
        self.max_n_obj_per_class = max_n_obj_per_class if max_n_obj_per_class else 1e10
        self.current_idx = {d: 0 for d in self.data_subsets}
        self._load_images_paths()
        self.classes = {
            s: sorted(list(set(self.images_labels[s]))) for s in self.data_subsets}
        self.n_classes = {s: len(self.classes[s]) for s in self.data_subsets}
        self.n_samples = {d: len(self.images_paths[d]) for d in self.data_subsets}
        self.indexes = {d: {cl: np.where(np.array(self.images_labels[d]) == cl)[
            0] for cl in self.classes[d]} for d in self.data_subsets}

    def _load_images_paths(self):
        skip_list = ['train','val','test']
        n_files_dataset = 0
        n_files_selected = 0
        n_classes_selected = 0
        random.seed(4)
        
        for d in self.data_subsets:
            print('{:5} ======================================'.format(d))
            self.images_paths[d] = []
            self.images_labels[d] = []
            for root, dirs, files in os.walk(self.dataset_path+d):
                files_filtered = [f for f in files if (f.endswith('.jpg') or f.endswith('.png') and not f.startswith('._'))]
                n_obj = len(files_filtered)
                n_files_dataset+=n_obj
                curr_class = root.split('/')[-1]

                if (n_obj<self.min_n_obj_per_class or n_obj>=self.max_n_obj_per_class)  and d == 'train':
                    skip_list.append(curr_class)
                    print('Class {:11} WAS SKIPPED'.format(curr_class))
                    continue
                    
                if curr_class in skip_list:
                    continue
                
                idx_list = list(range(n_obj))
                random.shuffle(idx_list)
                count = 0
                for i in idx_list:
                    if count >= self.select_max_n_obj_per_class:
                        break
                    f = files_filtered[i]
                    self.images_paths[d].append(root+'/'+f)
                    self.images_labels[d].append(curr_class)
                    count+=1
                n_files_selected+=count
                if d == 'train':
                    n_classes_selected += 1
                print('Class {:11}: total number of files {:6}, selected {:6}'.format(curr_class, n_obj, count))
        print('Total number of files in dataset: {}'.format(n_files_dataset))
        print('Number of selected files: {}'.format(n_files_selected))
        print('Number of selected classes: {}'.format(n_classes_selected))
        print('Number of skipped classes: {}'.format(len(skip_list)))

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

    def get_batch_random(self, batch_size,  s='train'):
        images = [np.zeros((batch_size, self.input_shape[0], self.input_shape[1], 3))]
        targets = []

        selected_classes_idxs = random.sample(range(self.n_classes[s]), batch_size)
        count = 0

        for i in list(selected_classes_idxs):
            curr_class = self.classes[s][i]
            curr_class_n_elements = len(self.indexes[s][curr_class])
            indx = np.random.randint(curr_class_n_elements, size=1)
            
            imgs = self._get_images_set(
                [curr_class], [indx], s=s, with_aug=False)
            images[0][count,:,:,:] = imgs[0]
            targets.append(curr_class)
            count += 1
        return images, targets

    def get_batch(self, batch_size,  s='train'):
        images = [
            np.zeros((batch_size, self.input_shape[0], self.input_shape[1], 3))]
        targets = np.zeros((batch_size, self.n_classes[s]))

        count = 0
        with_aug = self.augmentations
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

        with_aug = self.augmentations
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

            with_aug = self.augmentations
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

        with_aug = self.augmentations
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

    def generate(self, batch_size, is_binary=False, mode='siamese', s='train'):
        while True:
            if mode == 'siamese':
                data, targets = self.get_batch_pairs(batch_size, s)
            elif mode == 'triplet':
                data, targets = self.get_batch_triplets(batch_size, s)
            else:
                data, targets = self.get_batch(batch_size, s)
                if is_binary:
                    targets = targets[:,0]
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


    def plot_batch_simple(self, data, targets):
        num_imgs = data[0].shape[0]
        img_h = data[0].shape[1]
        img_w = data[0].shape[2]
        full_img = np.zeros((img_h,num_imgs*img_w,3), dtype=np.uint8)
        indxs = np.argmax(targets, axis=1)
        class_names = [self.classes['train'][i] for i in indxs]
        
        for i in range(num_imgs):
            full_img[:,i*img_w:(i+1)*img_w,:] = data[0][i,:,:,:]
            cv2.putText(full_img, class_names[i], (img_w*i + 10, 20), cv2.FONT_HERSHEY_SIMPLEX,  
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        plt.figure(figsize = (20,2))
        plt.imshow(full_img)
        plt.show()

    
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