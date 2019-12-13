import albumentations as A


def get_aug(name='default', input_shape=[48, 48, 3]):
    if name == 'default':
        augmentations = A.Compose([
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(p=0.4),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=30, val_shift_limit=30, p=0.4),
            A.CLAHE(p=0.4),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(blur_limit=1, p=0.3),
            A.GaussNoise(var_limit=(50, 80), p=0.3),
            A.RandomCrop(p=1, height=input_shape[1], width=input_shape[0])
        ], p=1)
    else:
        augmentations = None

    return augmentations
