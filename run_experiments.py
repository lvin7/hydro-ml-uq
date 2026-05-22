import sys, os, json, itertools, argparse
import time
import shutil
import gc
import joblib
import tensorflow as tf
from keras.layers import LSTM, Dense
from keras import backend as K
from tcn import TCN, tcn_full_summary
from tkan import TKAN
from keras.initializers import Orthogonal

# Disable auto-JIT and very aggressive autotuning on GPU
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

from data_utils import data_prep
from hp_tuning import hp_tuner, full_train
from metrics import plot_loss, scatter_plot, scatter_plot_1dah, metrics_table

def reset_tuner_directory(path):
    """Delete the tuner directory if corrupted."""
    if os.path.exists(path):
        print(f"[WARN] Removing corrupted tuner directory: {path}")
        shutil.rmtree(path)
    else:
        print(f"[WARN] Tuner directory not found: {path}")


def model_type(s):
    lookup = {
        "LSTM": LSTM,
        "TCN": TCN,
        "TKAN": TKAN,
        "Dense": Dense,
    }
    try:
        return lookup[s]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Unknown model: {s}")
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", type=model_type, default=[LSTM, TCN, TKAN, Dense])
    ap.add_argument("--nwps", nargs="+", default=['ifs', 'ukmo', 'gfs', 'gem'])
    ap.add_argument("--features", nargs="+", default=['Qp', 'Qpt', 'Qpts', 'Qptsd'])
    ap.add_argument("--tuners", nargs="+", default=['bayesian', 'random', 'hyperband', 'evol'])
    ap.add_argument("--lags", nargs="+", type=int, default=[2, 3, 4, 5])
    #---------------------------------------------------
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--epochs_fast", type=int, default=100)
    ap.add_argument("--epochs_full", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="models") # change to final-models later
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
            #lag=lag, # lag is not used, instead we use replicates to see variability in training
            val_start=args.val_start,
            test_start=args.test_start,
        )
        input_shape = X_train[0].shape
        horizon = y_train.shape[1]
        try:
            best_model, best_hp = hp_tuner(X_train, y_train, X_val, y_val, 
                                       input_shape, horizon, model_arch, tuner_name, 
                                       trials=args.trials, epochs=args.epochs_fast, output_dir=run_dir)
        except AttributeError as e:
            if "tolist" in str(e):
                print("\n[ERROR] DE tuner checkpoint corrupted.")
                print("[ACTION] Deleting tuner directory and restarting the tuner...\n")

                reset_tuner_directory(run_dir)
                raise RuntimeError(
                    f"DE tuner checkpoint corrupted at {run_dir}. "
                    "Deleted the tuner directory and rerunning..."
                )
            raise  

        total_time = time.time() - start_time

        best_model.save(os.path.join(run_dir, "best_model.keras"))
        with open(os.path.join(run_dir, "best_model_config.json"), "w") as f:
            json.dump(best_model.get_config(), f, indent=2)
        summary_path = os.path.join(run_dir, "tuning_summary.json")
        if os.path.exists(summary_path):
            print(f"[INFO] Summary already exists at {summary_path}.")
        else:
            with open(summary_path, "w") as f:
                json.dump({"best_score": best_hp.values, "total_tuning_time(min)": total_time / 60}, f, indent=2)

        fname = f"model-{model_name}"
        f"_nwp-{nwp}"
        f"_feat-{feats}"
        f"_lag-{lag}"
        f"_tuner-{tuner_name}"

        model_path   = os.path.join(run_dir, f"{fname}.keras")

        if os.path.exists(model_path):
            print(f"[INFO] Final model already exists at {model_path}. Skipping final training.")
        else:
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

            # Save model
            model.save(model_path)
            # Save scalers
            joblib.dump(scalers, os.path.join(run_dir, f"{fname}.scalers.pkl"))
            # Save metrics
            metrics.to_csv(os.path.join(run_dir, f"{fname}.csv"), index=False)

        # Clean up
        print("🧹 Cleaning memory...")

        # Clear TensorFlow / Keras session
        try:
            K.clear_session()
        except Exception:
            pass

        # delete created objects
        for obj in ["best_model", "best_hp", "tuner"]:
            try:
                del globals()[obj]
            except KeyError:
                try:
                    del locals()[obj]   # works in interactive shells, not in functions
                except:
                    pass

        # forced Garbage Collection
        gc.collect()
        time.sleep(0.5)

if __name__ == "__main__":
    main()