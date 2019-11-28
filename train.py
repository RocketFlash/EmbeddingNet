import os
import numpy as np
from embedding_net.model import EmbeddingNet
from embedding_net.pretrain_backbone_softmax import pretrain_backbone_softmax
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


n_epochs = 1000
n_steps_per_epoch = 200
val_batch_size = 8
val_steps = 200

config_name = 'road_signs_resnet18_max80_min30'
config_file_name = 'configs/{}.yml'.format(config_name)
model = EmbeddingNet(config_file_name)

pretrain_backbone_softmax(model, config_file_name)

initial_lr = 1e-4
decay_factor = 0.99
step_size = 1

callbacks = [
    LearningRateScheduler(lambda x: initial_lr *
                          decay_factor ** np.floor(x/step_size)),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1),
    EarlyStopping(patience=5, verbose=1),
    TensorBoard(log_dir=model.tensorboard_log_path),
    ModelCheckpoint(filepath=os.path.join(model.weights_save_path, model.model_save_name),
                    verbose=1, monitor='val_loss', save_best_only=True)
]

model.train_generator_mining(steps_per_epoch=n_steps_per_epoch, 
                             epochs=n_epochs,
                             callbacks = callbacks, 
                             val_steps=val_steps, 
                             n_classes=5, 
                             n_samples=3,
                             negative_selection_mode='semihard')

model.generate_encodings(save_file_name='encodings_{}.pkl'.format(config_name),
                         max_num_samples_of_each_classes=30, knn_k=1, shuffle=True)

model_accuracy = model.calculate_prediction_accuracy()
print('Model accuracy on validation set: {}'.format(model_accuracy))
