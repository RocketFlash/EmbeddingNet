import keras.backend as K
from model import SiameseNet
from data_loader import SiameseImageLoader
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import Model, load_model
import albumentations as A
import cv2
import time


model = SiameseNet('configs/road_signs.yml')
model.load_model('{}best_model_4.h5'.format(model.weights_save_path))
model.load_encodings('{}encodings.pkl'.format(model.encodings_path))


model_accuracy = model.calculate_prediction_accuracy()
print('Model accuracy on validation set: {}'.format(model_accuracy))
