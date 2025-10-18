import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.callbacks import EarlyStopping

from models import build_model

def choose_hp(hp):
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
    "lr": hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling="LOG"),
    "wd": hp.Float('wd', min_value=0.0, max_value=0.1, step=0.01),
    "cn": hp.Float('cn', min_value=0.0, max_value=5.0, step=0.5),
    "quantile": hp.Float('quantile', min_value=0.3, max_value=0.8, step=0.05),
    "activation": hp.Choice('activation', values=["tanh", "relu", "elu"]),
    }

    return layer_hp, global_hp


# Use Keras Tuner to choose the best hyperparameters
def hp_tuning(X_train, y_train, X_val, y_val):
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=10,
        directory=".",
        project_name="hp_tuning"
    )
    tuner.search(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True)
        ]
        )
    best_hp = tuner.get_best_hyperparameters()[0]
    return best_hp