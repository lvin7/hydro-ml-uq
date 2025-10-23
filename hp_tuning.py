import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.layers import LSTM
from tcn import TCN, tcn_full_summary
from tkan import TKAN
import ast

from models import RestoreBestMeanLoss, build_model


class MyHyperModel(kt.HyperModel):
    '''
    Inherit from the keras_tuner HyperModel class. Tweak to enable different model_arch and horizons as input.
    '''
    def __init__(self, input_shape, horizon, model_arch=LSTM, *, name=None, tunable=True):
        super().__init__(name=name, tunable=tunable)
        self.input_shape = input_shape
        self.horizon = horizon
        self.model_arch = model_arch

    def build(self, hp):
        # Define search space
        if self.model_arch == LSTM:
            num_layers = hp.Choice('num_layers', values=[1, 2, 3])
            layer_hp = {
                "LSTM": {
                    "units_l0": hp.Choice('units_l0', values=[16, 32, 64, 128, 256]), 
                    "units_l1": hp.Choice('units_l1', values=[16, 32, 64, 128, 256]), 
                    "units_l2": hp.Choice('units_l2', values=[16, 32, 64, 128, 256]),
                    "recurrent_dropout": hp.Float('recurrent_dropout', min_value=0.0, max_value=0.5, step=0.05),
                    "layer_norm": hp.Boolean('layer_norm'),
                    "bidir": hp.Boolean('bidir'),
                }
            }
        elif self.model_arch == TCN:
            num_layers = 1
            layer_hp={
                "TCN": {
                    "nb_filters": hp.Choice('nb_filters', values=[8, 16, 32, 64, 128]),
                    "kernel_size": hp.Choice('kernel_size', values=[2, 3, 5]),
                    "nb_stacks": hp.Choice('nb_stacks', values=[1, 2]),
                    "dilations": ast.literal_eval(hp.Choice('dilations', values=['[1, 2, 4]', '[1, 2, 4, 8]', '[1, 2, 4, 8, 16]'])),
                    "kernel_initializer": hp.Choice('kernel_initializer', values=['he_normal', 'glorot_uniform']),
                    "layer_norm": hp.Boolean('layer_norm'),
                },
            }
        elif self.model_arch == TKAN:
            num_layers = hp.Choice('num_layers', values=[1, 2, 3])
            layer_hp = {
                "TKAN": {
                    "units_l0": hp.Choice('units_l0', values=[32, 64, 128, 256]),
                    "units_l1": hp.Choice('units_l1', values=[32, 64, 128, 256]),
                    "units_l2": hp.Choice('units_l2', values=[32, 64, 128, 256]),
                    "layer_norm": hp.Boolean('layer_norm')
                }        
            }
        else:
            print('No model found.')

        global_hp = {
        "num_layers": num_layers,
        "dropout_rate": hp.Float('dropout_rate', min_value=0.05, max_value=0.7, step=0.05),
        "lr": hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling='log'),
        "wd": hp.Float('wd', min_value=0.0, max_value=0.1, step=0.01),
        "cn": hp.Float('cn', min_value=0.0, max_value=5.0, step=0.5),
        "quantile": hp.Float('quantile', min_value=0.3, max_value=0.8, step=0.05),
        "activation": hp.Choice('activation', values=["tanh", "relu", "elu"]),
        }
        # Call build_model function
        model = build_model(self.input_shape, self.horizon, self.model_arch, global_hp, layer_hp)
        '''
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
        '''
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        kwargs["batch_size"] = kwargs.get("batch_size", hp.Choice("batch_size", [16, 32, 64, 128]))
        return model.fit(*args, **kwargs)

# Use Keras Tuner to choose the best hyperparameters
def hp_tuner(X_train, y_train, X_val, y_val, 
             input_shape, horizon, model_arch, tuner_type, 
             trials=50, epochs=50, output_dir='hp_tuning'):
    
    if tuner_type == 'random':
        tuner = kt.RandomSearch(
        MyHyperModel(input_shape, horizon, model_arch),
        objective=kt.Objective('val_r2', direction='max'),
        max_trials=trials,
        directory=output_dir,
        project_name="rand"
        )
    elif tuner_type == 'bayes':
        tuner = kt.BayesianOptimization(
        MyHyperModel(input_shape, horizon, model_arch),
        objective=kt.Objective('val_mae', direction='min'),
        max_trials=trials,
        directory=output_dir,
        project_name="bayes"
        )
    elif tuner_type == 'hyperband':
        tuner = kt.Hyperband(
        MyHyperModel(input_shape, horizon, model_arch),
        objective=kt.Objective('val_mae', direction='min'),
        max_epochs=trials,
        factor=3,
        directory=output_dir,
        project_name="hb"
        )
    else:
        print('Yikes, not an adequate tuner.')
    
    tuner.search(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=epochs,
        shuffle=False,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True)
        ]
        )
    tuner.search_space_summary()
    # Get best model
    best_models = tuner.get_best_models(num_models=1)
    #best_model[0].summary()
    # Get best hyperparams
    best_hp = tuner.get_best_hyperparameters()[0]
    return best_models[0], best_hp

# We can use this to fully retrain the best model
def full_train(best_hp, X_train, y_train, X_val, y_val, input_shape, horizon=5, model_arch=LSTM, epochs=200, patience=30):
    hypermodel = MyHyperModel(input_shape, horizon, model_arch)
    model = hypermodel.build(best_hp)
    
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(5, patience//2), cooldown=2, verbose=1)
    restore_best = RestoreBestMeanLoss(patience=patience, start_from_epoch=10)
    print(X_train.shape, y_train.shape)
    history = hypermodel.fit(best_hp, model, X_train, y_train, 
                             validation_data=(X_val, y_val), # change later so val is included in training (some left for early-stopping)
                             epochs=epochs,
                             shuffle=False,
                             callbacks=[reduce_lr, restore_best, TerminateOnNaN()],
    )
    best_val = np.min(history.history['val_loss'])
    return model, history, best_val
    