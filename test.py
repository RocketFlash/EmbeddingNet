from model import SiameseNet

model = SiameseNet('SiameseNet/configs/road_signs.yml')
model.load_model('{}best_model_4.h5'.format(model.weights_save_path))
model.load_encodings('{}encodings.pkl'.format(model.encodings_path))


model_accuracy = model.calculate_prediction_accuracy()
print('Model accuracy on validation set: {}'.format(model_accuracy))
