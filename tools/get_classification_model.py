from tensorflow.keras.models import load_model, Model
import efficientnet.tfkeras as efn
from embedding_net.utils import parse_params
from embedding_net.model_new import SiameseNet

model_path = '/home/rauf/EmbeddingNet/work_dirs/deepfake_efn_b3/weights/best_deepfake_efn_b3_001_0.578046.hdf5'
cfg_params = parse_params('configs/deepfake_siamese.yml')

model = SiameseNet(cfg_params, training=True)
model.model.load_weights(model_path, by_name=True)
# model.load_model(model_path)

model_classification = model.classification_model
model_classification.save("model_classification.h5")
