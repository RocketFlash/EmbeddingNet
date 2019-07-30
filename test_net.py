from model import SiameseNet
from data_loader import SiameseImageLoader
import matplotlib.pyplot as plt
from keras import optimizers
import albumentations as A


def plot_grapth(values, y_label, title):
    t = list(range(len(values)))
    fig, ax = plt.subplots()
    ax.plot(t, values)

    ax.set(xlabel='iteration', ylabel='{}'.format(y_label),
           title='{}'.format(title))
    ax.grid()

    fig.savefig("plots/{}.png".format(y_label))


n_epochs = 100
n_iterations = 500

augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.4),
    A.RandomGamma(p=0.4),
    A.HueSaturationValue(hue_shift_limit=20,
                         sat_shift_limit=50, val_shift_limit=50, p=0.4),
    A.CLAHE(p=0.4),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Blur(blur_limit=11, p=0.3),
    A.GaussNoise(var_limit=(100, 150), p=0.3),
    A.CenterCrop(p=1, height=256, width=256)
], p=1)

loader = SiameseImageLoader(
    '/home/rauf/plates_competition/dataset/to_train/', input_shape=(256, 256, 3), augmentations=augmentations)


optimizer = optimizers.Adam(lr=1e-5)
# optimizer = optimizers.RMSprop(lr=1e-5)
model = SiameseNet(input_shape=(256, 256, 3), backbone='resnet18', mode='l2',
                   image_loader=loader, optimizer=optimizer)

train_losses, train_accuracies, val_losses, val_accuracies = model.train(
    steps_per_epoch=20, epochs=20)

plot_grapth(train_losses, 'train_loss', 'Losses on train')
plot_grapth(train_accuracies, 'train_acc', 'Accuracies on train')
plot_grapth(val_losses, 'val_loss', 'Losses on val')
plot_grapth(val_accuracies, 'val_acc', 'Accuracies on val')
