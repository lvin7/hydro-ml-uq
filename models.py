import tensorflow as tf
import numpy as np
from keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LayerNormalization, InputLayer, Bidirectional
from keras.metrics import R2Score
from keras.initializers import Orthogonal

# Use a seed with the Orthogonal initializer for TKAN
#initializer = Orthogonal(seed=1)  --- what's this for?

# Restore best mean loss callback
class RestoreBestMeanLoss(Callback):
    def __init__(self, patience=0, start_from_epoch=0):
        super(RestoreBestMeanLoss, self).__init__()
        self.patience = patience
        self.start_from_epoch = start_from_epoch
        self.best_weights = None
        self.best_mean_loss = np.inf
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the mean loss (train + validation)
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if train_loss is not None and val_loss is not None:
            mean_loss = (train_loss + val_loss*2) / 3
        else:
            return  # Skip if one of the losses is missing
        # Check if we should update best weights
        if epoch >= self.start_from_epoch:
            if mean_loss < self.best_mean_loss:
                self.best_mean_loss = mean_loss
                self.best_weights = self.model.get_weights()
                self.wait = 0  # Reset patience counter
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f'Restored model weights from epoch {self.stopped_epoch - self.patience + 1} with best mean loss {self.best_mean_loss}')


class PinballLoss(tf.keras.losses.Loss):
    def __init__(self, quantile, name='pinball_loss'):
        super().__init__(name=name)
        self.quantile = quantile

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(self.quantile * error, (self.quantile - 1) * error))
    

def build_model(input_shape, horizon, model_arch=LSTM, global_hp={}, layer_hp={}):
    """
    Build and compile an ML model for multi-step forecasting.
    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        horizon (int): Forecast horizon.
        model_arch (class): Keras layer class to use (e.g., LSTM, TCN).
        hp (dict): Hyperparameters for the model (global).
        layer_hp (dict): Hyperparameters specific to the layer architecture.
    General model builder for multi-step forecasting - works for LSTM, GRU, TCN, TKAN etc.
    """
    model_name = model_arch.__name__
    if len(layer_hp.values()) > 0:
        layer_hp = layer_hp[model_name]
    # Build model
    model = Sequential(name=f'{model_name}_model')
    model.add(InputLayer(shape=input_shape))
    for i in range(global_hp.get('num_layers', 1)):
        if model_name != 'TCN':
            units = layer_hp.get(f'units_l{i}', 128)
            
        kwargs = {k: v for k, v in layer_hp.items() if not k.startswith('units_') and k not in ['bidir', 'layer_norm']}
        kwargs['return_sequences'] = True if i < global_hp.get('num_layers', 1) - 1 else False
        kwargs.setdefault('activation', global_hp.get('activation', 'relu'))
        if model_name != 'TCN':
            cell = model_arch(units, **kwargs) # Add hidden layer
        else:
            cell = model_arch(**kwargs) # Add hidden layer
        if layer_hp.get('bidir', False):
            cell = Bidirectional(cell)
        model.add(cell)
        model.add(Dropout(global_hp.get('dropout_rate', 0.2)))
        if layer_hp.get('layer_norm', True):
            model.add(LayerNormalization())
    model.add(Dense(horizon))
    # Compile the model with Pinball loss
    optimizer = tf.keras.optimizers.AdamW(learning_rate=global_hp.get('lr', 0.001), weight_decay=global_hp.get('wd', 0.0), clipnorm=global_hp.get('cn', 1.0))
    r2_metric = R2Score(name='r2') 
    model.compile(optimizer=optimizer, loss=PinballLoss(quantile=global_hp.get('quantile', 0.5)), metrics=['mae'])
    model.summary()
    return model


# train_full function is used in hp_tuning.py, this is excess
def train_model(model, X_train, y_train, X_val, y_val, epochs=500, batch_size=32, patience=20):
    """
    Train the model with early stopping and restore best mean loss.
    Args:
        model (tf.keras.Model): The compiled Keras model.
        X_train (np.array): Training input data.
        y_train (np.array): Training target data.
        X_val (np.array): Validation input data.
        y_val (np.array): Validation target data.
        epochs (int): Maximum number of epochs to train.
        batch_size (int): Batch size for training.
        patience (int): Patience for early stopping.
    Returns:
        history: Training history object.
    """
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(5, patience//2), cooldown=2, verbose=1)
    restore_best = RestoreBestMeanLoss(patience=patience, start_from_epoch=10)
    print(X_train.shape, y_train.shape)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[reduce_lr, restore_best, TerminateOnNaN()],
    )
    best_val = np.min(history.history['val_loss'])
    return history, best_val


def load_model(path):
    model = tf.keras.load_model(path)
    # add special calls for TCN and TKAN
    return model