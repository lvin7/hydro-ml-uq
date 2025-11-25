import sys, os, json, itertools, argparse
import time
import joblib
import tensorflow as tf
from keras.layers import LSTM
from tcn import TCN, tcn_full_summary
from tkan import TKAN
from keras.initializers import Orthogonal

from data_utils import data_prep
from hp_tuning import hp_tuner, full_train
from metrics import plot_loss, scatter_plot, scatter_plot_1dah, metrics_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[LSTM])
    ap.add_argument("--nwps", nargs="+", default=['ifs'])
    ap.add_argument("--features", nargs="+", default=['Qpt', 'Qpts', 'Qptsd'])
    ap.add_argument("--tuners", nargs="+", default=['bayesian', 'random', 'hyperband', 'evol'])
    ap.add_argument("--lags", nargs="+", type=int, default=[2, 3, 4, 5])
    #---------------------------------------------------
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--epochs_fast", type=int, default=100)
    ap.add_argument("--epochs_full", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="models") # change to models later
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--val_start", type=str, default='2023-01-01')
    ap.add_argument("--test_start", type=str, default='2023-10-01')
    args = ap.parse_args()

    tf.random.set_seed(args.seed)

    # Loop through the selected options
    grid = itertools.product(args.models, args.nwps, args.features, args.lags, args.tuners)

    # Main code
    for (model_arch, nwp, feats, lag, tuner_name) in grid:
        model_name = model_arch.__name__
        print(f"\n=== {model_name} | {nwp} | {feats} | lag={lag} | tuner={tuner_name} ===")
        run_dir = os.path.join(args.outdir, model_name, f"nwp={nwp}", f"feat={feats}", f"lag={lag}", tuner_name)
        os.makedirs(run_dir, exist_ok=True)

        print("Running experiments...")
        start_time = time.time()

        X_train, y_train, X_val, y_val, X_test, y_test, scalers = data_prep(
            nwp=nwp,
            target='Q',
            vars=feats,
            val_start=args.val_start,
            test_start=args.test_start,
        )
        input_shape = X_train[0].shape
        horizon = y_train.shape[1]
        best_model, best_hp = hp_tuner(X_train, y_train, X_val, y_val, 
                                       input_shape, horizon, model_arch, tuner_name, 
                                       trials=args.trials, epochs=args.epochs_fast, output_dir=run_dir)

        total_time = time.time() - start_time

        best_model.save(os.path.join(run_dir, "best_model.keras"))
        with open(os.path.join(run_dir, "best_model_config.json"), "w") as f:
            json.dump(best_model.get_config(), f, indent=2)
        with open(os.path.join(run_dir, "tuning_summary.json"), "w") as f:
            json.dump({"best_score": best_hp.values, "total_tuning_time(min)": total_time / 60}, f, indent=2)

        print("Training the final model...")
        model, history, best_val = full_train(
            best_hp, X_train, y_train, X_val, y_val, input_shape, horizon=args.horizon, model_arch=model_arch, epochs=args.epochs_full, patience=30
        )
        y_pred = model.predict(X_test)
        metrics = metrics_table(y_pred, y_test)
        plot_loss(history, save_path=run_dir)
        print(best_val)
        scatter_plot(y_pred, y_test, model_name=model_name, save_path=run_dir)
        scatter_plot_1dah(y_pred, y_test, model_name=model_name, save_path=run_dir)

        fname = f"model-{model_name}"
        f"_nwp-{nwp}"
        f"_feat-{feats}"
        f"_lag-{lag}"
        f"_tuner-{tuner_name}"

        # Save model
        model.save(os.path.join(run_dir, f"{fname}.keras"))

        # Save scalers
        joblib.dump(scalers, os.path.join(run_dir, f"{fname}.scalers.pkl"))

        # Save metrics
        metrics.to_csv(os.path.join(run_dir, f"{fname}.csv"), index=False)

if __name__ == "__main__":
    main()