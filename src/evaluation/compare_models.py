import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score

# TensorFlow import is optional — compare_models works even if TF is not installed,
# it will simply skip the Neural Network entry if the .keras file is missing.
try:
    from tensorflow import keras as _keras
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    """Load project configuration to get model thresholds."""
    if not CONFIG_PATH.exists():
        print("[WARN] Config file not found. Using default threshold 0.5")
        return {"model": {"thresholds": {}, "threshold": 0.5}}
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        # Ensure thresholds dictionary exists
        if "model" not in config:
            config["model"] = {}
        if "thresholds" not in config["model"]:
            config["model"]["thresholds"] = {}
        return config


def evaluate_model(model_path, X, y, model_name="Model", threshold=None):
    """
    Evaluates a joblib-serialised sklearn/XGBoost/RF model.
    If threshold is provided, uses it for predictions.
    """
    if not model_path.exists():
        print(f"[WARN] {model_name} not found at {model_path.name}. Skipping.")
        return None

    try:
        model = joblib.load(model_path)

        # Get probabilities (needed for AUC and Custom Threshold)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = np.zeros(len(y))
            if threshold is not None:
                print(f"[WARN] {model_name} does not support probabilities. Ignoring threshold.")
                threshold = None

        # Generate Predictions
        if threshold is not None:
            # Apply Custom Threshold
            y_pred = (y_prob >= threshold).astype(int)
        else:
            # Default Model Behavior (usually 0.5)
            y_pred = model.predict(X)

        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred),
            "F1-Score": f1_score(y, y_pred),
            "ROC-AUC": roc_auc_score(y, y_prob),
        }
    except Exception as e:
        print(f"[ERROR] Failed to evaluate {model_name}: {e}")
        return None


def evaluate_nn_model(model_path, X, y, model_name="Neural Network", threshold=0.5):
    """
    Evaluates a Keras Neural Network model stored in the .keras format.

    Keras models are NOT joblib-compatible — they must be loaded with
    keras.models.load_model().  This function mirrors evaluate_model() but
    handles the TF-specific loading and prediction interface.

    Parameters
    ----------
    model_path : Path
        Path to the .keras model file.
    X : np.ndarray
        Feature matrix (already preprocessed / scaled).
    y : array-like
        True binary labels.
    threshold : float
        Decision threshold for converting sigmoid output to {0, 1}.
    """
    if not _TF_AVAILABLE:
        print("[WARN] TensorFlow not available. Skipping Neural Network evaluation.")
        return None

    if not model_path.exists():
        print(f"[WARN] {model_name} not found at {model_path.name}. Skipping.")
        return None

    try:
        model = _keras.models.load_model(str(model_path))

        # predict() returns shape (n_samples, 1) for a single-output sigmoid
        y_prob = model.predict(X, verbose=0).ravel()
        y_pred = (y_prob >= threshold).astype(int)

        return {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred),
            "F1-Score": f1_score(y, y_pred),
            "ROC-AUC": roc_auc_score(y, y_prob),
        }
    except Exception as e:
        print(f"[ERROR] Failed to evaluate {model_name}: {e}")
        return None


def run_comparison():
    # 1. Load Validation Data
    try:
        val_df = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
        X_val = val_df.drop(columns=["target"])
        y_val = val_df["target"]
    except FileNotFoundError:
        print("Error: Validation data not found.")
        return

    # 2. Get Model Thresholds from Config
    config = load_config()
    thresholds_dict = config.get("model", {}).get("thresholds", {})
    default_threshold = config.get("model", {}).get("threshold", 0.5)

    print("[INFO] Using model-specific thresholds from Config")
    print(f"[INFO] Default threshold: {default_threshold}")
    if thresholds_dict:
        print(f"[INFO] Model thresholds: {thresholds_dict}")

    # 3. Collect Metrics with Model-Specific Thresholds
    all_metrics = {}

    # --- A. Baseline & Phase 2 Models (Model-Specific Thresholds) ---
    baseline_threshold = thresholds_dict.get("baseline_logreg", default_threshold)
    baseline_metrics = evaluate_model(
        MODELS_DIR / "baseline_logreg.pkl",
        X_val,
        y_val,
        f"Baseline (Thresh={baseline_threshold:.3f})",
        threshold=baseline_threshold,
    )
    if baseline_metrics:
        all_metrics["Baseline"] = baseline_metrics

    rf_threshold = thresholds_dict.get("random_forest_model_smote", default_threshold)
    rf_metrics = evaluate_model(
        MODELS_DIR / "random_forest_model_smote.pkl",
        X_val,
        y_val,
        f"Random Forest (Thresh={rf_threshold:.3f})",
        threshold=rf_threshold,
    )
    if rf_metrics:
        all_metrics["Random Forest"] = rf_metrics

    xgb_threshold = thresholds_dict.get("xgboost_model_smote", default_threshold)
    xgb_metrics = evaluate_model(
        MODELS_DIR / "xgboost_model_smote.pkl",
        X_val,
        y_val,
        f"XGBoost (Manual) (Thresh={xgb_threshold:.3f})",
        threshold=xgb_threshold,
    )
    if xgb_metrics:
        all_metrics["XGBoost (Manual)"] = xgb_metrics

    opt_threshold = thresholds_dict.get("xgboost_optimized", default_threshold)
    opt_metrics = evaluate_model(
        MODELS_DIR / "xgboost_optimized.pkl",
        X_val,
        y_val,
        f"XGBoost (Optimized) (Thresh={opt_threshold:.3f})",
        threshold=opt_threshold,
    )
    if opt_metrics:
        all_metrics["XGBoost (Optimized)"] = opt_metrics

    weighted_threshold = thresholds_dict.get("xgboost_weighted", default_threshold)
    weighted_metrics = evaluate_model(
        MODELS_DIR / "xgboost_weighted.pkl",
        X_val,
        y_val,
        f"XGBoost (Weighted) (Thresh={weighted_threshold:.3f})",
        threshold=weighted_threshold,
    )
    if weighted_metrics:
        all_metrics["XGBoost (Weighted)"] = weighted_metrics

    # --- B. Champion Model (Optimized Threshold) ---
    champion_threshold = thresholds_dict.get("xgboost_weighted_optimized", default_threshold)
    opt_weighted_metrics = evaluate_model(
        MODELS_DIR / "xgboost_weighted_optimized.pkl",
        X_val,
        y_val,
        f"Champion (Thresh={champion_threshold:.3f})",
        threshold=champion_threshold,
    )
    if opt_weighted_metrics:
        all_metrics["Champion"] = opt_weighted_metrics

    # --- C. Neural Network baseline (Phase 4) ---
    nn_threshold = thresholds_dict.get("neural_network_model", default_threshold)
    nn_metrics = evaluate_nn_model(
        MODELS_DIR / "neural_network_model.keras",
        X_val.values,  # convert DataFrame -> numpy for Keras
        y_val.values,
        model_name=f"Neural Network (Thresh={nn_threshold:.3f})",
        threshold=nn_threshold,
    )
    if nn_metrics:
        all_metrics["Neural Network"] = nn_metrics

    # --- D. Neural Network Optuna-tuned (Phase 4 improved) ---
    nn_opt_threshold = thresholds_dict.get("neural_network_optuna", default_threshold)
    nn_opt_metrics = evaluate_nn_model(
        MODELS_DIR / "neural_network_optuna.keras",
        X_val.values,
        y_val.values,
        model_name=f"NN Optuna (Thresh={nn_opt_threshold:.3f})",
        threshold=nn_opt_threshold,
    )
    if nn_opt_metrics:
        all_metrics["NN Optuna"] = nn_opt_metrics

    # 4. Generate DataFrame and Plot
    if not all_metrics:
        print("No models found.")
        return

    # Dynamic DataFrame Construction
    first_metric = list(all_metrics.values())[0]
    data = {"Metric": list(first_metric.keys())}
    for model_name, metrics in all_metrics.items():
        data[model_name] = list(metrics.values())

    comparison_df = pd.DataFrame(data)

    print("\n--- Final Model Comparison Table ---")
    print(comparison_df.round(7))

    # Plot
    df_melted = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")

    plt.figure(figsize=(14, 7))

    # 1. Capture the axes object 'ax'
    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")

    # Get champion threshold for title
    champion_threshold = thresholds_dict.get("xgboost_weighted_optimized", default_threshold)
    plt.title(
        (
            f"Model Comparison with Model-Specific Thresholds "
            f"(Champion Thresh={champion_threshold:.3f})"
        ),
        fontsize=14,
        fontweight="bold",
    )
    plt.ylim(0, 1.15)  # Increased slightly to make room for text
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    plt.grid(axis="y", alpha=0.3)

    # 2. Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    plt.tight_layout()

    save_path = RESULTS_DIR / "comparison_production_final.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[SUCCESS] Final comparison plot saved to {save_path}")


if __name__ == "__main__":
    run_comparison()
