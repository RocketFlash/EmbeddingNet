import albumentations as A


def get_aug(name='default', input_shape=[48, 48, 3]):
    if name == 'default':
        augmentations = A.Compose([
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(p=0.4),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=30, val_shift_limit=30, p=0.4),
            A.CLAHE(p=0.4),
            A.Blur(blur_limit=1, p=0.3),
            A.GaussNoise(var_limit=(50, 80), p=0.3)
        ], p=1)
    elif name == 'plates':
        augmentations = A.Compose([
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(p=0.4),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=30, 
                                 val_shift_limit=30, 
                                 p=0.4),
            A.CLAHE(p=0.4),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(blur_limit=1, p=0.3),
            A.GaussNoise(var_limit=(50, 80), p=0.3),
            A.RandomCrop(p=0.8, height=2*input_shape[1]/3, width=2*input_shape[0]/3)
        ], p=1)
    elif name == 'deepfake':
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
        ], p=1)
    elif name == 'plates2':
        augmentations = A.Compose([
            A.CLAHE(clip_limit=(1,4),p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightness(limit=0.2, p=0.3),
            A.RandomContrast(limit=0.2, p=0.3),
            # A.Rotate(limit=360, p=0.9),
            A.RandomRotate90(p=0.3),
            A.HueSaturationValue(hue_shift_limit=(-50,50), 
                                 sat_shift_limit=(-15,15), 
                                 val_shift_limit=(-15,15), 
                                 p=0.5),
#             A.Blur(blur_limit=(5,7), p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.CenterCrop(p=1, height=2*input_shape[1]//3, width=2*input_shape[0]//3),
            A.Resize(p=1, height=input_shape[1], width=input_shape[0])
        ], p=1)
    else:
        augmentations = None

    return augmentations
