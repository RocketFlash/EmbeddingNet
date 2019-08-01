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


dataset_path = '/home/rauf/plates_competition/dataset/to_train/'
n_epochs = 20
n_steps_per_epoch = 600
batch_size = 16
val_steps = 100

augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.4),
    A.RandomGamma(p=0.4),
    A.HueSaturationValue(hue_shift_limit=20,
                         sat_shift_limit=50, val_shift_limit=50, p=0.4),
    A.CLAHE(p=0.4),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Blur(blur_limit=5, p=0.3),
    A.GaussNoise(var_limit=(100, 150), p=0.3),
    A.CenterCrop(p=1, height=256, width=256)
], p=1)

loader = SiameseImageLoader(dataset_path, input_shape=(
    256, 256, 3), augmentations=augmentations)


optimizer = optimizers.Adam(lr=1e-5)
# optimizer = optimizers.RMSprop(lr=1e-5)
model = SiameseNet(input_shape=(256, 256, 3), backbone='resnet50', mode='l2',
                   image_loader=loader, optimizer=optimizer)

train_losses, train_accuracies, val_losses, val_accuracies = model.train(
    steps_per_epoch=n_steps_per_epoch, val_steps=val_steps, epochs=n_epochs)

plot_grapth(train_losses, 'train_loss', 'Losses on train')
plot_grapth(train_accuracies, 'train_acc', 'Accuracies on train')
plot_grapth(val_losses, 'val_loss', 'Losses on val')
plot_grapth(val_accuracies, 'val_acc', 'Accuracies on val')


model.generate_encodings()
# model.load_encodings('encodings/encodings.pkl')
prediction = model.predict(
    '/home/rauf/plates_competition/dataset/test/0000.jpg')
print(prediction)
