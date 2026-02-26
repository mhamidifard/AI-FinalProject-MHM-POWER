"""
optimize_neural_network.py
===========================
Phase 4 (Improved): Optuna-tuned Neural Network for Bank Marketing Binary Classification.

WHY THIS SCRIPT EXISTS — Diagnosis of F1=0.47
----------------------------------------------
The initial NN (train_neural_network.py) underperformed XGBoost because:

1. NO HYPERPARAMETER SEARCH — XGBoost champion ran 60 Optuna trials; the NN used
   hand-picked defaults. Correct fix: Optuna search over architecture + regularisation.

2. NO THRESHOLD TUNING — NN used threshold=0.5. Imbalanced datasets (~89/11%) need
   a lower threshold (~0.25-0.40) to maximise F1. Correct fix: search over thresholds.

3. OVER-PARAMETERISED ARCHITECTURE — 256→128→64 = ~42K params for 31K samples
   causes overfitting even with dropout. Correct fix: smaller search space (64→32).

4. WRONG EARLY STOPPING METRIC — val_auc maximisation ≠ val_f1 maximisation for
   imbalanced data. AUC is threshold-agnostic; F1 is threshold-dependent and punishes
   the 0.5 threshold harder on imbalanced sets. Correct fix: monitor val_loss or use
   a custom F1 callback, then sweep threshold post-training.

Approach
--------
* Optuna (same framework as XGBoost champion) tunes: # layers, units/layer, dropout,
  L2 lambda, learning rate, and batch size. 30 trials with 3-fold lightweight CV.
* Post-best-trial: retrain on full training set, then sweep thresholds [0.2, 0.5]
  on the validation set to maximise F1, exactly mirroring tune_threshold.py.
* Final evaluation on the held-out TEST set with the optimal threshold.

Conventions
-----------
Mirrors optimize_weighted_xgboost.py: Optuna study → train best model → evaluate →
plot → save. WandB logging via wandb_utils.py.
"""

import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from pathlib import Path

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.training.wandb_utils import (
    init_wandb,
    log_metrics,
    log_config,
    log_artifact,
    log_image,
    log_confusion_matrix,
    finish_wandb,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Optuna: silence per-trial verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)
# Suppress TF per-epoch output during CV trials (restored for final training)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# Hyper-parameter search bounds
N_OPTUNA_TRIALS: int = 30        # ≥30 gives reliable results; 50 for production
N_CV_FOLDS: int = 3              # Fast CV inside trials; full train after
MAX_EPOCHS_TRIAL: int = 40       # Short budget per trial — EarlyStopping handles it
MAX_EPOCHS_FINAL: int = 200      # Full budget for the best retrained model
EARLY_STOP_PATIENCE: int = 8     # Patience for trials (tighter = faster search)
FINAL_EARLY_STOP_PATIENCE: int = 20  # Patience for final training


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data() -> Tuple[np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray,
                          int]:
    """
    Load preprocessed CSV splits. Returns numpy arrays + input_dim.
    The preprocessor has already: imputed, scaled numerics, OHE categoricals.
    """
    print("[INFO] Loading preprocessed data...")
    try:
        train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
        val_df   = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
        test_df  = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data not found in '{DATA_PROCESSED_DIR}'. "
            "Run 'python -m src.preprocessing.main' first."
        )

    X_train = train_df.drop(columns=["target"]).values.astype(np.float32)
    y_train = train_df["target"].values

    X_val   = val_df.drop(columns=["target"]).values.astype(np.float32)
    y_val   = val_df["target"].values

    X_test  = test_df.drop(columns=["target"]).values.astype(np.float32)
    y_test  = test_df["target"].values

    input_dim = X_train.shape[1]

    print(f"      Train : {X_train.shape} | Positive rate: {y_train.mean()*100:.1f}%")
    print(f"      Val   : {X_val.shape}")
    print(f"      Test  : {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, input_dim


# =============================================================================
# 2. CLASS WEIGHT
# =============================================================================

def get_class_weight(y: np.ndarray) -> Dict[int, float]:
    """Balanced class weights to handle the ~89/11 imbalance."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw: Dict[int, float] = dict(zip(classes.astype(int), weights))
    print(f"[INFO] Class weights: {cw}")
    return cw


# =============================================================================
# 3. MODEL BUILDER (called once per trial and for the final model)
# =============================================================================

def build_model(input_dim: int, params: dict) -> keras.Model:
    """
    Builds a NN with architecture controlled by the params dict.
    Config keys used:
        n_layers     : number of hidden layers (1–3)
        units_l{i}   : neurons in layer i
        dropout_l{i} : dropout rate in layer i
        l2_lambda    : L2 regularisation coefficient
    """
    l2_reg = regularizers.L2(float(params["l2_lambda"]))

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    for i in range(int(params["n_layers"])):
        units   = int(params[f"units_l{i}"])
        dropout = float(params[f"dropout_l{i}"])

        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=l2_reg,
            name=f"dense_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="NN_Optuna")


# =============================================================================
# 4. OPTUNA OBJECTIVE — trains a small model and returns mean CV F1
# =============================================================================

def objective(trial: optuna.Trial,
              X_train: np.ndarray,
              y_train: np.ndarray,
              input_dim: int) -> float:
    """
    Optuna objective function. Builds + trains a NN for a given set of
    hyperparameters and returns mean stratified-CV validation F1.

    Search Space
    ------------
    Architecture:
        n_layers : 1–3
        units    : 32, 64, 128, 256 per layer
        dropout  : 0.1–0.5 per layer
        l2       : log-uniform [1e-5, 1e-2]
    Training:
        lr       : log-uniform [1e-4, 1e-2]
        batch    : categorical {128, 256, 512}

    CV threshold for F1: swept [0.25, 0.50] (5 values) independently per fold,
    taking the best per-fold threshold — this gives a ceiling estimate of
    achievable F1 without data leakage.
    """
    # ── Architecture search space ─────────────────────────────────────────────
    n_layers = trial.suggest_int("n_layers", 1, 3)
    params: dict = {
        "n_layers": n_layers,
        "l2_lambda": trial.suggest_float("l2_lambda", 1e-5, 1e-2, log=True),
    }
    for i in range(n_layers):
        params[f"units_l{i}"]   = trial.suggest_categorical(f"units_l{i}",   [32, 64, 128, 256])
        params[f"dropout_l{i}"] = trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)

    lr         = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    # ── Class weight ──────────────────────────────────────────────────────────
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw      = dict(zip(classes.astype(int), weights))

    # ── Stratified CV ─────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_f1s: list = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = build_model(input_dim, params)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=MAX_EPOCHS_TRIAL,
            batch_size=batch_size,
            class_weight=cw,
            callbacks=callbacks,
            verbose=0,             # silent during trials
        )

        # Sweep thresholds to find per-fold best F1
        y_prob = model.predict(X_va, verbose=0).ravel()
        best_fold_f1 = 0.0
        for thresh in np.linspace(0.20, 0.50, 7):
            y_pred = (y_prob >= thresh).astype(int)
            fold_f1 = f1_score(y_va, y_pred, zero_division=0)
            if fold_f1 > best_fold_f1:
                best_fold_f1 = fold_f1

        fold_f1s.append(best_fold_f1)

        # Prune unpromising trials early (Optuna median pruner)
        trial.report(best_fold_f1, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Free memory
        del model
        tf.keras.backend.clear_session()

    return float(np.mean(fold_f1s))


# =============================================================================
# 5. THRESHOLD TUNING (post-training, on validation set)
# =============================================================================

def find_best_threshold(model: keras.Model,
                         X_val: np.ndarray,
                         y_val: np.ndarray) -> float:
    """
    Sweeps classification thresholds on the VALIDATION set and returns the
    threshold that maximises binary F1-Score.

    WHY on validation (not train)?
    Using val for threshold selection is standard practice — it mirrors
    tune_threshold.py and ensures no test-set leakage.
    """
    y_prob = model.predict(X_val, verbose=0).ravel()

    best_thresh: float = 0.5
    best_f1: float = 0.0
    thresholds = np.linspace(0.15, 0.60, 46)  # 1% steps

    print("\n[INFO] Sweeping thresholds on validation set...")
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    print(f"      Best threshold : {best_thresh:.3f}")
    print(f"      Best val F1    : {best_f1:.4f}")
    return best_thresh


# =============================================================================
# 6. TRAIN FINAL MODEL with best Optuna params
# =============================================================================

def train_final_model(best_params: dict,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       input_dim: int,
                       class_weight_dict: Dict[int, float]) -> Tuple[keras.Model, object]:
    """
    Retrain on the full training set using the best params found by Optuna.
    Uses longer epochs + more patience than trial runs.
    """
    lr:         float = float(best_params["learning_rate"])
    batch_size: int   = int(best_params["batch_size"])

    print("\n[INFO] Training final model with best hyperparameters...")
    model = build_model(input_dim, best_params)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=FINAL_EARLY_STOP_PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS_FINAL,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n[INFO] Final training finished at epoch {len(history.epoch)}.")
    return model, history


# =============================================================================
# 7. EVALUATE
# =============================================================================

def evaluate_model(model: keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   threshold: float) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate on the truly held-out test set using the tuned threshold.
    """
    print(f"\n[INFO] Evaluating on Test Set (threshold={threshold:.3f})...")
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "Accuracy":  float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_test, y_pred)),
        "F1-Score":  float(f1_score(y_test, y_pred)),
        "ROC-AUC":   float(roc_auc_score(y_test, y_prob)),
    }

    print("\n" + "=" * 62)
    print("NEURAL NETWORK (OPTUNA-TUNED) — TEST SET EVALUATION")
    print("=" * 62)
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")
    print("=" * 62)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

    # WandB
    log_metrics({
        "test_accuracy":  metrics["Accuracy"],
        "test_precision": metrics["Precision"],
        "test_recall":    metrics["Recall"],
        "test_f1_score":  metrics["F1-Score"],
        "test_roc_auc":   metrics["ROC-AUC"],
        "best_threshold": threshold,
    })
    log_confusion_matrix(y_test, y_pred, class_names=["No", "Yes"])

    return metrics, y_pred, y_prob


# =============================================================================
# 8. PLOTS
# =============================================================================

def plot_results(history: object,
                 y_test: np.ndarray,
                 y_pred: np.ndarray,
                 y_prob: np.ndarray,
                 save_dir: Path) -> None:
    """Training curves, confusion matrix, ROC curve."""

    # ── Training curves ───────────────────────────────────────────────────────
    h = history.history  # type: ignore[union-attr]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Neural Network (Optuna-Tuned) — Training History",
                 fontsize=14, fontweight="bold")

    axes[0].plot(h["loss"],     label="Train Loss",  color="#2196F3")
    axes[0].plot(h["val_loss"], label="Val Loss",    color="#F44336", linestyle="--")
    axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(h["auc"],     label="Train AUC", color="#4CAF50")
    axes[1].plot(h["val_auc"], label="Val AUC",   color="#FF9800", linestyle="--")
    axes[1].set_title("AUC per Epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = save_dir / "nn_optuna_training_curves.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log_image(str(p), "training_curves_optuna")
    print(f"[INFO] Training curves → {p}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred: No", "Pred: Yes"],
                yticklabels=["True: No", "True: Yes"])
    plt.title("Confusion Matrix — NN Optuna (Test Set)", fontsize=13, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout()
    p = save_dir / "confusion_matrix_nn_optuna.png"
    plt.savefig(p, dpi=300); plt.close()
    log_image(str(p), "confusion_matrix_optuna")
    print(f"[INFO] Confusion matrix → {p}")

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#9C27B0", linewidth=2,
             label=f"NN Optuna (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.fill_between(fpr, tpr, alpha=0.08, color="#9C27B0")
    plt.xlabel("False Positive Rate", fontsize=12); plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — NN Optuna (Test Set)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11); plt.grid(alpha=0.3); plt.tight_layout()
    p = save_dir / "roc_curve_nn_optuna.png"
    plt.savefig(p, dpi=300); plt.close()
    log_image(str(p), "roc_curve_optuna")
    print(f"[INFO] ROC curve → {p}")


# =============================================================================
# 9. MAIN
# =============================================================================

if __name__ == "__main__":
    init_wandb(
        run_name="neural-network-optuna-tuned",
        tags=["neural-network", "tensorflow", "optuna", "phase-4", "improved"],
    )

    log_config({
        "model_type": "NeuralNetwork_Optuna",
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "n_cv_folds": N_CV_FOLDS,
        "framework": f"TensorFlow {tf.__version__}",
        "random_seed": RANDOM_SEED,
        "improvement_over": "train_neural_network.py",
    })

    try:
        # ── 1. Load Data ──────────────────────────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test, input_dim = load_data()

        # ── 2. Class Weights ──────────────────────────────────────────────────
        class_weight_dict = get_class_weight(y_train)

        # ── 3. Optuna Study ───────────────────────────────────────────────────
        print(f"\n[INFO] Starting Optuna search ({N_OPTUNA_TRIALS} trials, "
              f"{N_CV_FOLDS}-fold CV)...")
        print("       This will take several minutes. Each trial = short NN training.\n")

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study  = optuna.create_study(direction="maximize", pruner=pruner)

        def wandb_callback(study: optuna.Study, trial: optuna.Trial) -> None:
            """Log each trial's result to WandB."""
            if trial.state == optuna.trial.TrialState.COMPLETE:
                log_metrics({
                    "trial_f1":     trial.value,
                    "trial_number": trial.number,
                })

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, input_dim),
            n_trials=N_OPTUNA_TRIALS,
            callbacks=[wandb_callback],
            show_progress_bar=True,
        )

        print("\n" + "=" * 62)
        print("OPTUNA OPTIMIZATION RESULTS")
        print("=" * 62)
        print(f"  Best CV F1    : {study.best_value:.4f}")
        print(f"  Best trial #  : {study.best_trial.number}")
        print("  Best params   :")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")
        print("=" * 62)

        log_metrics({"best_optuna_cv_f1": study.best_value})
        log_config({"best_params": study.best_params})

        # ── 4. Train Final Model ──────────────────────────────────────────────
        model, history = train_final_model(
            best_params=study.best_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
            class_weight_dict=class_weight_dict,
        )

        # ── 5. Tune Threshold on Validation Set ───────────────────────────────
        best_threshold = find_best_threshold(model, X_val, y_val)

        # ── 6. Evaluate on Test Set ───────────────────────────────────────────
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, best_threshold)

        # ── 7. Plots ──────────────────────────────────────────────────────────
        plot_results(history, y_test, y_pred, y_prob, RESULTS_DIR)

        # ── 8. Save Model ─────────────────────────────────────────────────────
        save_path = MODELS_DIR / "neural_network_optuna.keras"
        model.save(save_path)
        print(f"\n[INFO] Optuna-tuned model saved → {save_path}")
        log_artifact(str(save_path), artifact_name="nn_optuna", artifact_type="model")

        print("\n[SUCCESS] Optuna NN pipeline completed.")
        print(f"          Model   → {save_path}")
        print(f"          Charts  → {RESULTS_DIR}")
        print(f"          Threshold → {best_threshold:.3f}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise

    finally:
        finish_wandb()
