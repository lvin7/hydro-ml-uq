import sys
from data_utils import data_prep
from models import build_model

def main():
    # Main code
    print("Running experiments...")
    X_train, y_train, X_val, y_val, X_test, y_test, scalers = data_prep(
        nwp='ifs',
        target='Q',
        vars='Qpt',
    )
    build_model(input_shape=X_train[0].shape, horizon=5)
    pass

if __name__ == "__main__":
    main()