import sys
from keras.layers import LSTM

from data_utils import data_prep
from models import build_model, train_model, save_model
from hp_tuning import hp_tuner

def main(
        horizon=5,
        model_arch=LSTM,
):
    # Main code
    print("Running experiments...")
    X_train, y_train, X_val, y_val, X_test, y_test, scalers = data_prep(
        nwp='ifs',
        target='Q',
        vars='Qpt',
    )
    input_shape = X_train[0].shape
    print(input_shape)
    pass
    best_hp = hp_tuner(X_train, y_train, X_val, y_val, input_shape, horizon, model_arch)
    print(best_hp)
    print(best_hp.shape)
    #model = build_model(input_shape=X_train[0].shape, horizon=5)
    #_, best_val = train_model(model, X_train, y_train, X_val, y_val)
    #save_model(model)
    #print(best_val)
    pass

if __name__ == "__main__":
    main()