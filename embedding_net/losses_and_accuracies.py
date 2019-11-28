import keras.backend as K
import keras


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def triplet_loss(margin=0.5):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    def loss_function(y_true, y_pred):
        total_lenght = y_pred.shape.as_list()[-1]

        anchor = y_pred[:, 0:int(total_lenght*1/3)]
        positive = y_pred[:, int(total_lenght*1/3):int(total_lenght*2/3)]
        negative = y_pred[:, int(total_lenght*2/3):int(total_lenght*3/3)]

        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor-positive), axis=1)

        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor-negative), axis=1)

        # compute loss
        basic_loss = pos_dist-neg_dist+margin
        loss = K.maximum(basic_loss, 0.0)
        return loss

    return loss_function


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


class tSNECallback(keras.callbacks.Callback):

    def __init__(self, save_file_name='tSNE.gif'):
        super(tSNECallback, self).__init__()
        self.save_file_name = save_file_name

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
