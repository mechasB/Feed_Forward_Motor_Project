# src/functions/evaluation.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error


def _compute_metrics(model_name, y_true, y_pred, inference_time):
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()

    r2    = r2_score(y_true_np, y_pred_np)
    rmse  = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae   = mean_absolute_error(y_true_np, y_pred_np)
    medae = median_absolute_error(y_true_np, y_pred_np)

    mask = y_true_np != 0
    mape = np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100 if mask.any() else float('nan')

    ms_per_sample = (inference_time / len(y_true_np)) * 1000
    freq_hz       = 1000 / ms_per_sample if ms_per_sample > 0 else float('inf')

    return {
        'model':         model_name,
        'R2':            r2,
        'RMSE [W]':      rmse,
        'MAE [W]':       mae,
        'MedAE [W]':     medae,
        'MAPE [%]':      mape,
        'ms/próbkę':     ms_per_sample,
        'Częst. [Hz]':   freq_hz,
    }


def display_model_results(
    model_name,
    y_true,
    y_pred,
    inference_time,
    history=None,
    history_metric='loss',
    history_val_key=None,
    feature_importances=None,
    feature_names=None,
    subset_size=300,
):
    """
    Ujednolicone wyświetlanie wyników modelu.

    Parametry
    ---------
    model_name        : str   – nazwa modelu (do tytułów)
    y_true            : array – rzeczywiste wartości mocy [W]
    y_pred            : array – przewidywane wartości mocy [W]
    inference_time    : float – łączny czas predykcji na zbiorze testowym [s]
    history           : dict  – historia treningu (np. model.history.history lub evals_result())
                                Obsługiwane formaty:
                                  Keras:   {'loss': [...], 'val_loss': [...]}
                                  XGBoost: {'validation_0': {'rmse': [...]}, 'validation_1': {...}}
    history_metric    : str   – klucz metryki w history (domyślnie 'loss')
    history_val_key   : str   – klucz danych walidacyjnych (domyślnie 'val_<history_metric>')
    feature_importances: array – wartości ważności cech (opcjonalne)
    feature_names     : list  – nazwy cech (opcjonalne, wymagane gdy feature_importances != None)
    subset_size       : int   – liczba próbek w wykresie czasowym (domyślnie 300)

    Zwraca
    ------
    dict – słownik metryk (można przekazać do compare_models)
    """
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    residuals = y_true_np - y_pred_np

    metrics = _compute_metrics(model_name, y_true_np, y_pred_np, inference_time)

    # --- WYDRUK METRYK ---
    sep = "=" * 52
    print(sep)
    print(f"  WYNIKI MODELU: {model_name}")
    print(sep)
    print(f"  R2 Score      : {metrics['R2']:.4f}   (im bliżej 1.0 tym lepiej)")
    print(f"  RMSE          : {metrics['RMSE [W]']:.2f} W")
    print(f"  MAE           : {metrics['MAE [W]']:.2f} W")
    print(f"  MedAE         : {metrics['MedAE [W]']:.2f} W")
    print(f"  MAPE          : {metrics['MAPE [%]']:.2f} %")
    print("-" * 52)
    print(f"  Czas/próbkę   : {metrics['ms/próbkę']:.4f} ms")
    print(f"  Częstotliwość : {metrics['Częst. [Hz]']:.1f} Hz")
    print(sep + "\n")

    # --- UKŁAD WYKRESÓW ---
    has_history      = history is not None
    has_importances  = feature_importances is not None and feature_names is not None
    n_extra          = int(has_history) + int(has_importances)
    n_cols           = 2
    n_rows           = 2 + int(np.ceil(n_extra / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    fig.suptitle(f'Ewaluacja modelu: {model_name}', fontsize=18, fontweight='bold', y=1.01)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # WYKRES 1 – Scatter: Rzeczywistość vs Predykcja
    ax = axes[0]
    ax.scatter(y_true_np, y_pred_np, alpha=0.25, color='steelblue', edgecolors='none', s=10)
    lo, hi = min(y_true_np.min(), y_pred_np.min()), max(y_true_np.max(), y_pred_np.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Ideał (y=x)')
    ax.set_title('1. Korelacja: Rzeczywistość vs Predykcja', fontsize=13)
    ax.set_xlabel('Rzeczywista moc [W]')
    ax.set_ylabel('Przewidywana moc [W]')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    # WYKRES 2 – Przebieg czasowy
    ax = axes[1]
    n = min(subset_size, len(y_true_np))
    ax.plot(y_true_np[:n], label='Prawdziwa moc', color='black', linewidth=1.5, alpha=0.8)
    ax.plot(y_pred_np[:n], label='Predykcja', color='lime', linestyle='--', linewidth=1.5)
    ax.set_title(f'2. Śledzenie sygnału (pierwsze {n} próbek)', fontsize=13)
    ax.set_xlabel('Numer próbki')
    ax.set_ylabel('Moc [W]')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

    # WYKRES 3 – Reszty vs Predykcja
    ax = axes[2]
    ax.scatter(y_pred_np, residuals, alpha=0.2, color='mediumpurple', edgecolors='none', s=10)
    ax.axhline(0, color='red', linestyle='-', lw=2)
    ax.set_title('3. Analiza reszt (gdzie model się myli?)', fontsize=13)
    ax.set_xlabel('Przewidywana moc [W]')
    ax.set_ylabel('Błąd [W]')
    ax.grid(True, linestyle=':', alpha=0.6)

    # WYKRES 4 – Histogram błędów
    ax = axes[3]
    q_lo = np.percentile(residuals, 0.5)
    q_hi = np.percentile(residuals, 99.5)
    clipped = residuals[(residuals > q_lo) & (residuals < q_hi)]
    sns.histplot(clipped, bins=60, ax=ax, color='darkorange', kde=True)
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero')
    ax.axvline(np.median(residuals), color='blue', linestyle=':', lw=2,
               label=f'Mediana: {np.median(residuals):.1f} W')
    ax.set_title('4. Rozkład błędu [W]', fontsize=13)
    ax.set_xlabel('Błąd [W]')
    ax.set_ylabel('Liczba próbek')
    ax.legend(fontsize=9)

    next_ax = 4

    # WYKRES 5 – Krzywa uczenia (opcjonalne)
    if has_history:
        ax = axes[next_ax]
        next_ax += 1
        _plot_learning_curve(ax, history, history_metric, history_val_key)

    # WYKRES 6 – Feature importance (opcjonalne)
    if has_importances:
        ax = axes[next_ax]
        next_ax += 1
        _plot_feature_importance(ax, feature_importances, feature_names)

    # Ukryj nieużywane osie
    for i in range(next_ax, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    return metrics


def _plot_learning_curve(ax, history, metric_key, val_key):
    """Obsługuje formaty Keras i XGBoost eval_result."""
    # Format XGBoost: {'validation_0': {'rmse': [...]}, 'validation_1': {'rmse': [...]}}
    if isinstance(history, dict) and all(isinstance(v, dict) for v in history.values()):
        keys = list(history.keys())
        # Szukamy metryki w zagnieżdżonym słowniku
        inner_keys = list(list(history.values())[0].keys())
        m = metric_key if metric_key in inner_keys else inner_keys[0]
        ax.plot(history[keys[0]][m], label=f'Trening ({m.upper()})')
        if len(keys) > 1:
            ax.plot(history[keys[1]][m], '--', label=f'Walidacja ({m.upper()})')
        ax.set_xlabel('Iteracje (drzewa)')
    else:
        # Format Keras: {'loss': [...], 'val_loss': [...]}
        if metric_key in history:
            ax.plot(history[metric_key], label=f'Trening ({metric_key})')
        v_key = val_key or f'val_{metric_key}'
        if v_key in history:
            ax.plot(history[v_key], '--', label=f'Walidacja ({v_key})')
        ax.set_xlabel('Epoka')

    ax.set_title('5. Krzywa uczenia', fontsize=13)
    ax.set_ylabel('Błąd')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)


def _plot_feature_importance(ax, importances, names):
    importances = np.array(importances)
    names       = np.array(names)
    idx         = np.argsort(importances)[-15:]  # top 15
    ax.barh(names[idx], importances[idx], color='teal')
    ax.set_title('6. Ważność cech (Feature Importance)', fontsize=13)
    ax.set_xlabel('Ważność')
    ax.grid(True, linestyle=':', alpha=0.4, axis='x')


def compare_models(results_list, sort_by='R2', ascending=False):
    """
    Wyświetla tabelę porównawczą dla listy wyników zwróconych przez display_model_results.

    Parametry
    ---------
    results_list : list[dict]  – lista słowników z metrykami (zwracanych przez display_model_results)
    sort_by      : str         – kolumna do sortowania (domyślnie 'R2')
    ascending    : bool        – kierunek sortowania

    Zwraca
    ------
    pd.DataFrame – tabela z wynikami
    """
    df = pd.DataFrame(results_list).set_index('model')
    df = df.sort_values(sort_by, ascending=ascending)

    fmt = {
        'R2':           '{:.4f}',
        'RMSE [W]':     '{:.2f}',
        'MAE [W]':      '{:.2f}',
        'MedAE [W]':    '{:.2f}',
        'MAPE [%]':     '{:.2f}',
        'ms/próbkę':    '{:.4f}',
        'Częst. [Hz]':  '{:.1f}',
    }

    print("\n" + "=" * 70)
    print("  PORÓWNANIE MODELI")
    print("=" * 70)
    print(df.to_string(float_format=lambda x: f'{x:.3f}'))
    print("=" * 70 + "\n")

    # Wykres słupkowy najważniejszych metryk
    metrics_to_plot = ['R2', 'RMSE [W]', 'MAE [W]', 'MedAE [W]', 'MAPE [%]', 'ms/próbkę']
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    fig.suptitle('Porównanie modeli', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for i, col in enumerate(metrics_to_plot):
        ax = axes[i]
        bars = ax.bar(df.index, df[col], color=colors)
        ax.set_title(col, fontsize=12)
        ax.set_ylabel(col)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    return df


# --- Wsteczna kompatybilność (stare funkcje) ---

def print_standardized_metrics(model_name, y_true, y_pred, inference_time):
    """Wyświetla ujednolicone metryki dla modelu (uproszczona wersja)."""
    m = _compute_metrics(model_name, y_true, y_pred, inference_time)
    print(f"=== Wyniki ewaluacji: {model_name} ===")
    print(f"R2 Score: {m['R2']:.4f}")
    print(f"RMSE:     {m['RMSE [W]']:.4f} W")
    print(f"MAE:      {m['MAE [W]']:.4f} W")
    print(f"MedAE:    {m['MedAE [W]']:.4f} W")
    print(f"Szybkość: {m['ms/próbkę']:.4f} ms/próbkę")
    print("=========================================\n")


def plot_standardized_results(model_name, y_true, y_pred, subset_size=200):
    """Generuje ujednoliconą siatkę 4 wykresów ewaluacyjnych (uproszczona wersja)."""
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    residuals = y_true_np - y_pred_np

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Ewaluacja Wizualna Modelu: {model_name}', fontsize=20, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    axes[0, 0].scatter(y_true_np, y_pred_np, alpha=0.3, color='blue', edgecolor='none')
    min_val = min(y_true_np.min(), y_pred_np.min())
    max_val = max(y_true_np.max(), y_pred_np.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_title('1. Korelacja: Rzeczywistość vs Predykcja', fontsize=14)
    axes[0, 0].set_xlabel('Rzeczywista Moc [W]')
    axes[0, 0].set_ylabel('Przewidywana Moc [W]')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    axes[0, 1].plot(y_true_np[:subset_size], label='Prawdziwa Moc', color='black', linewidth=2)
    axes[0, 1].plot(y_pred_np[:subset_size], label='Predykcja', color='lime', linestyle='--', linewidth=2)
    axes[0, 1].set_title(f'2. Śledzenie sygnału (pierwsze {subset_size} próbek)', fontsize=14)
    axes[0, 1].set_xlabel('Numer próbki w czasie')
    axes[0, 1].set_ylabel('Moc [W]')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    axes[1, 0].scatter(y_pred_np, residuals, alpha=0.3, color='purple', edgecolor='none')
    axes[1, 0].axhline(0, color='red', linestyle='-', lw=2)
    axes[1, 0].set_title('3. Analiza Reszt (Gdzie model się myli?)', fontsize=14)
    axes[1, 0].set_xlabel('Przewidywana Moc [W]')
    axes[1, 0].set_ylabel('Błąd (Rzeczywistość - Predykcja) [W]')
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    q_low = np.percentile(residuals, 0.5)
    q_hi  = np.percentile(residuals, 99.5)
    filtered_residuals = residuals[(residuals > q_low) & (residuals < q_hi)]
    sns.histplot(filtered_residuals, bins=50, ax=axes[1, 1], color='orange', kde=True)
    axes[1, 1].axvline(0, color='red', linestyle='--', label='Błąd zero')
    axes[1, 1].set_title('4. Rozkład Błędu (w Watach)', fontsize=14)
    axes[1, 1].set_xlabel('Błąd [W]')
    axes[1, 1].set_ylabel('Liczba wystąpień')
    axes[1, 1].legend()

    plt.show()
