import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error

# -----------------------
# Metrics helper functions
# -----------------------

def plot_loss(history, ylim=None, save_path=''):
    plt.figure(figsize=(14, 7))
    plt.title('Model loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    if ylim!=None:
      plt.ylim(0, ylim)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(f'{save_path}/loss.png')


def scatter_plot(y_pred, y_true, model_name='_ Model', save_path=''):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true) & ~np.isinf(y_pred) & ~np.isinf(y_true)
    y_pred_clean = y_pred[mask].reshape(-1, y_pred.shape[1])
    y_obs_clean = y_true[mask].reshape(-1, y_true.shape[1])

    y_pred_flat = y_pred_clean.flatten()
    y_obs_flat = y_obs_clean.flatten()

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(f'{model_name} Scatter Plot', fontsize=16, y=0.8, x=0.4)
    ax_scatter = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)

    for i in range(5):
        ax_scatter.scatter(y_obs_clean[:, i], y_pred_clean[:, i],
                            label=f'{i + 1} day(s) ahead', alpha=0.3, s=25)

    ax_scatter.set_xlabel('Observed discharge (m³/s)', fontsize=14)
    ax_scatter.set_ylabel('Predicted discharge (m³/s)', fontsize=14)
    max_val = max(y_obs_clean.max(), y_pred_clean.max())
    ax_scatter.plot([0, max_val], [0, max_val], 'k-', label='45 degree line')

    model = LinearRegression()
    model.fit(y_obs_flat.reshape(-1, 1), y_pred_flat)
    predicted_regression = model.predict(y_obs_flat.reshape(-1, 1))
    ax_scatter.plot(y_obs_flat, predicted_regression, 'r--', lw=2, label='Mean Regression Line')
    ax_scatter.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{save_path}/scatter.png')
    plt.close()


def scatter_plot_1dah(y_pred, y_true, model_name='_Model', save_path=''):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true[:,0], y_pred[:,0], alpha=0.5)

    reg = LinearRegression().fit(y_true[:, [0]], y_pred[:, 0])
    y_line = reg.predict(y_true[:, [0]])

    plt.plot(y_true[:,0], y_line, 'r--', lw=2, label='Regression')
    plt.plot(y_true[:,0], y_true[:,0], 'k-', lw=1, label='1:1')
    plt.legend()
    plt.title(model_name)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{save_path}/scatter_1dah.png')
    plt.close()


def metrics_table(y_pred, y_true):
    threshold = np.percentile(y_true, 75)
    metrics = []

    for i in range(y_pred.shape[1]):
        forecast = y_pred[:, i]
        observed = y_true[:, i]
        mask = observed >= threshold
        observed_peaks = observed[mask]
        forecast_peaks = forecast[mask]

        full_metrics = {
            'Day Ahead': f'{i + 1} Day(s)',
            'MAE (Full)': mean_absolute_error(observed, forecast),
            'MAPE (Full)': mean_absolute_percentage_error(observed, forecast),
            'RMSE (Full)': np.sqrt(mean_squared_error(observed, forecast)),
            'Max Error (Full)': np.max(np.abs(observed - forecast)),
            'NSE (Full)': r2_score(observed, forecast)
        }

        peak_metrics = {
            'MAE (Peaks)': mean_absolute_error(observed_peaks, forecast_peaks),
            'MAPE (Peaks)': mean_absolute_percentage_error(observed_peaks, forecast_peaks),
            'RMSE (Peaks)': np.sqrt(mean_squared_error(observed_peaks, forecast_peaks)),
            'Max Error (Peaks)': np.max(np.abs(observed_peaks - forecast_peaks)),
            'NSE (Peaks)': r2_score(observed_peaks, forecast_peaks)
        }

        metrics.append({**full_metrics, **peak_metrics})

    return pd.DataFrame(metrics)
