import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau, TerminateOnNaN
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, LayerNormalization, InputLayer, Bidirectional
from tcn import TCN, tcn_full_summary
from tkan import TKAN


class MultiHyperModel(kt.HyperModel):
    def __init__(self, input_shape, horizon, model_arch=LSTM):
        self.input_shape = input_shape
        self.horizon = horizon
        self.model_arch = model_arch

    def build(self, hp):
        # Define search space
        layer_hp = {
        "LSTM": {
            "units_l0": hp.Choice('units_l0', values=[16, 32, 64, 128, 256]), 
            "units_l1": hp.Choice('units_l1', values=[16, 32, 64, 128, 256]), 
            "units_l2": hp.Choice('units_l2', values=[16, 32, 64, 128, 256]),
            "recurrent_dropout": hp.Float('recurrent_dropout', min_value=0.0, max_value=0.5, step=0.05),
            "layer_norm": hp.Boolean('layer_norm'),
            "bidir": hp.Boolean('bidir'),
        },
        "TCN": {
            "nb_filters": hp.Choice('nb_filters', values=[16, 32, 64, 128, 256]),
            "kernel_size": hp.Choice('kernel_size', values=[2, 3, 5]),
            "nb_stacks": hp.Choice('nb_stacks', values=[1, 2, 3]),
            "dilations": hp.Choice('dilations', values=[[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16]]),
            "kernel_initializer": hp.Choice('kernel_initializer', values=['glorot_uniform']),
            "layer_norm": hp.Boolean('layer_norm'),
        },
        "TKAN": {
            "units_l0": hp.Choice('units_l0', values=[32, 64, 128, 256]),
            "units_l1": hp.Choice('units_l1', values=[32, 64, 128, 256]),
            "units_l2": hp.Choice('units_l2', values=[32, 64, 128, 256]),
            "layer_norm": hp.Boolean('layer_norm')
        }
        }

        global_hp = {
        "num_layers": hp.Choice('num_layers', values=[1, 2, 3]),
        "dropout_rate": hp.Float('dropout_rate', min_value=0.05, max_value=0.7, step=0.05),
        "lr": hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling='log'),
        "wd": hp.Float('wd', min_value=0.0, max_value=0.1, step=0.01),
        "cn": hp.Float('cn', min_value=0.0, max_value=5.0, step=0.5),
        "quantile": hp.Float('quantile', min_value=0.3, max_value=0.8, step=0.05),
        "activation": hp.Choice('activation', values=["tanh", "relu", "elu"]),
        }

        # This part can be removed and build_model can be called....

        model_name = self.model_arch.__name__
        if len(layer_hp.values()) > 0:
            layer_hp = layer_hp[model_name]
        # Build model
        model = Sequential(name=f'{model_name}_model')
        model.add(InputLayer(shape=self.input_shape))
        for i in range(global_hp.get('num_layers', 1)):
            if model_name != 'TCN':
                units = layer_hp.get(f'units_l{i}', 128)
                
            kwargs = {k: v for k, v in layer_hp.items() if not k.startswith('units_') and k not in ['bidir', 'layer_norm']}
            if model_name != 'TCN':
                kwargs['return_sequences'] = True if i < global_hp.get('num_layers', 1) - 1 else False
                kwargs.setdefault('activation', global_hp.get('activation', 'relu'))
                cell = self.model_arch(units, **kwargs) # Add hidden layer
            else:
                cell = self.model_arch(**kwargs) # Add hidden layer
            if layer_hp.get('bidir', False):
                cell = Bidirectional(cell)
            model.add(cell)
            model.add(Dropout(global_hp.get('dropout_rate', 0.2)))
            if layer_hp.get('layer_norm', True):
                model.add(LayerNormalization())
        model.add(Dense(self.horizon))
        # Compile the model with Pinball loss
        optimizer = tf.keras.optimizers.AdamW(learning_rate=global_hp.get('lr', 0.001), weight_decay=global_hp.get('wd', 0.0), clipnorm=global_hp.get('cn', 1.0))
        model.compile(optimizer=optimizer, loss=PinballLoss(quantile=global_hp.get('quantile', 0.5)), metrics=['mae'])
        model.summary()

        return model

# Use Keras Tuner to choose the best hyperparameters
def hp_tuning(X_train, y_train, X_val, y_val, input_shape, horizon, model_arch):
    tuner = kt.RandomSearch(
        MultiHyperModel(input_shape, horizon, model_arch),
        objective="val_loss",
        max_trials=100,
        directory=".",
        project_name="hp_tuning"
    )
    tuner.search(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=kt.Int('batch_size', 16, 128, step=16, sampling='log'),
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True)
        ]
        )
    best_hp = tuner.get_best_hyperparameters(1)[0]
    return best_hp