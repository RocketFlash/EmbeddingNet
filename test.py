from siamese_net.model import SiameseNet

model = SiameseNet()
model.load_model('weights/road_signs/best_model_4.h5')
model.load_encodings('encodings/road_signs/encodings.pkl')

image_path = '/home/rauf/datasets/road_signs/road_signs_separated/val/1_1/rtsd-r1_train_006470.png'
model_prediction = model.predict(image_path)
print('Model prediction: {}'.format(model_prediction))
