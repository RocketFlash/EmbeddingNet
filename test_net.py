import os
import numpy as np
from model import SiameseNet
from data_loader import SiameseImageLoader
import matplotlib.pyplot as plt
from keras import optimizers
import albumentations as A
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def plot_grapth(values, y_label, title, project_name):
    t = list(range(len(values)))
    fig, ax = plt.subplots()
    ax.plot(t, values)

    ax.set(xlabel='iteration', ylabel='{}'.format(y_label),
           title='{}'.format(title))
    ax.grid()

    fig.savefig("plots/{}{}.png".format(project_name, y_label))


project_name = 'road_signs/'
dataset_path = '/home/rauf/plates_competition/dataset/road_signs/road_signs_separated/'
# project_name = 'plates/'
# dataset_path = '/home/rauf/plates_competition/dataset/to_train/'

n_epochs = 1000
n_steps_per_epoch = 500
batch_size = 4
val_steps = 100
input_shape = (48, 48, 3)
# input_shape = (256, 256, 3)

# augmentations = A.Compose([
#     A.RandomBrightnessContrast(p=0.4),
#     A.RandomGamma(p=0.4),
#     A.HueSaturationValue(hue_shift_limit=20,
#                          sat_shift_limit=50, val_shift_limit=50, p=0.4),
#     A.CLAHE(p=0.4),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.Blur(blur_limit=5, p=0.3),
#     A.GaussNoise(var_limit=(100, 150), p=0.3),
#     A.CenterCrop(p=1, height=256, width=256)
# ], p=0.5)

augmentations = None

loader = SiameseImageLoader(
    dataset_path, input_shape=input_shape, augmentations=augmentations)

optimizer = optimizers.Adam(lr=1e-4)
# optimizer = optimizers.RMSprop(lr=1e-5)
# model = SiameseNet(input_shape=(256, 256, 3), backbone='resnet50', mode='l2',
#                    image_loader=loader, optimizer=optimizer)

model = SiameseNet(input_shape=input_shape, backbone='simple2', backbone_weights='imagenet', mode='l2',
                   image_loader=loader, optimizer=optimizer, project_name=project_name,
                   freeze_backbone=False)


initial_lr = 1e-4
decay_factor = 0.99
step_size = 1

callbacks = [
    LearningRateScheduler(lambda x: initial_lr *
                          decay_factor ** np.floor(x/step_size)),
    EarlyStopping(patience=100, verbose=1),
    TensorBoard(log_dir=model.tensorboard_log_path),
    # ReduceLROnPlateau(factor=0.9, patience=50,
    #                   min_lr=1e-12, verbose=1),
    ModelCheckpoint(filepath=os.path.join(model.weights_save_path, 'best_model_2.h5'), verbose=1, monitor='loss',
                    save_best_only=True)
]

# train_losses, train_accuracies, val_losses, val_accuracies = model.train(
#     steps_per_epoch=n_steps_per_epoch, val_steps=val_steps, epochs=n_epochs)

H = model.train_generator(steps_per_epoch=n_steps_per_epoch, callbacks=callbacks,
                          val_steps=val_steps, epochs=n_epochs)
train_losses = H.history['loss']
train_accuracies = H.history['accuracy']
val_losses = H.history['val_loss']
val_accuracies = H.history['val_accuracy']

plot_grapth(train_losses, 'train_loss', 'Losses on train', project_name)
plot_grapth(train_accuracies, 'train_acc', 'Accuracies on train', project_name)
plot_grapth(val_losses, 'val_loss', 'Losses on val', project_name)
plot_grapth(val_accuracies, 'val_acc', 'Accuracies on val', project_name)


model.generate_encodings()
# model.load_encodings('encodings/encodings.pkl')
prediction = model.predict(
    '/home/rauf/plates_competition/dataset/road_signs/road_signs_separated/val/7_1/rtsd-r3_test_009188.png')
print(prediction)

model_accuracy = model.calculate_prediction_accuracy()
print('Model accuracy on validation set: {}'.format(model_accuracy))
