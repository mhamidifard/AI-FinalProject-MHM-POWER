"""
train_neural_network.py
=======================
Phase 4: Neural Network (TensorFlow/Keras) for Bank Marketing Binary Classification.

Architecture motivation
-----------------------
* Three hidden layers (256 → 128 → 64 neurons) with BatchNormalization + Dropout
  form a classic "funnel" that progressively compresses the tabular feature space
  into a compact classification representation.
* L2 weight regularization + Dropout work together to combat overfitting:
  - L2 penalises large weights during optimisation.
  - Dropout stochastically zeros neurons during training, forcing redundancy.
* Class-weight balancing (instead of SMOTE) lets the loss function natively
  penalise missed minority-class examples — a cleaner approach for NNs.
* EarlyStopping on val_auc (restored best weights) prevents overfitting by
  monitoring the metric that matters most for imbalanced classification.

Conventions
-----------
This file follows the exact same patterns as train_xgboost.py and train_rf.py:
  load_data() → build_model() → compile_and_train() → evaluate_model()
  → plot_results() → save_artifacts()
All console output uses the `[INFO] / [SUCCESS] / [ERROR]` prefix convention.
WandB logging uses the shared wandb_utils helpers.
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Scikit-learn utilities ────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

# ── WandB helpers (shared across all training scripts) ───────────────────────
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
# CONFIGURATION & CONSTANTS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "charts"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility seed — must be set before any TF graph is built
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Hyperparameters — typed constants (avoids Dict[str, Any] index type lints)
# Architecture
HIDDEN_UNITS: list = [256, 128, 64]       # neurons per hidden layer
DROPOUT_RATES: list = [0.40, 0.30, 0.20]  # matched to hidden layers
L2_LAMBDA: float = 1e-4                    # L2 weight-decay coefficient
# Training
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 256
MAX_EPOCHS: int = 200                      # EarlyStopping fires much sooner
EARLY_STOP_PATIENCE: int = 15             # epochs to wait before stopping
LR_REDUCE_PATIENCE: int = 7              # epochs before halving lr
LR_REDUCE_FACTOR: float = 0.5            # factor by which lr is reduced
# Decision threshold — tune later via tune_threshold.py
DECISION_THRESHOLD: float = 0.50

# Dict form used only for WandB config logging
HYPERPARAMS_LOG = {
    "hidden_units": HIDDEN_UNITS, "dropout_rates": DROPOUT_RATES,
    "l2_lambda": L2_LAMBDA, "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
    "early_stop_patience": EARLY_STOP_PATIENCE, "lr_reduce_patience": LR_REDUCE_PATIENCE,
    "lr_reduce_factor": LR_REDUCE_FACTOR, "threshold": DECISION_THRESHOLD,
}


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data() -> Tuple[np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray]:
    """
    Load the already-preprocessed CSV splits produced by preprocessing/main.py.

    The preprocessing pipeline has already:
      • Imputed missing values
      • StandardScaler-scaled numeric features
      • OneHotEncoded categorical features
      • Split data into train / val / test (70 / 15 / 15 %)

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test  (all numpy arrays)
    """
    print("[INFO] Loading preprocessed data...")

    try:
        train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
        val_df   = pd.read_csv(DATA_PROCESSED_DIR / "val.csv")
        test_df  = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"[ERROR] Data not found in '{DATA_PROCESSED_DIR}'. "
            "Please run 'python -m src.preprocessing.main' first."
        )

    # Separate features from the label column
    X_train, y_train = train_df.drop(columns=["target"]).values, train_df["target"].values
    X_val,   y_val   = val_df.drop(columns=["target"]).values,   val_df["target"].values
    X_test,  y_test  = test_df.drop(columns=["target"]).values,  test_df["target"].values

    # Cast to float32 — preferred dtype for TensorFlow operations
    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    print(f"      Train : {X_train.shape} | Positives: {y_train.sum()} "
          f"({y_train.mean()*100:.1f}%)")
    print(f"      Val   : {X_val.shape}   | Positives: {y_val.sum()} "
          f"({y_val.mean()*100:.1f}%)")
    print(f"      Test  : {X_test.shape}  | Positives: {y_test.sum()} "
          f"({y_test.mean()*100:.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# 2. CLASS-WEIGHT COMPUTATION (replaces SMOTE for Neural Networks)
# =============================================================================

def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights to handle the class imbalance (~89% / ~11%).

    WHY class weighting instead of SMOTE?
    NNs optimise their loss directly; weighting the loss per sample is
    mathematically equivalent to upsampling the minority class, without
    the artefacts that synthetic samples sometimes introduce.

    Returns a dict  {0: weight_for_majority, 1: weight_for_minority}
    that is passed to Keras model.fit(class_weight=...).
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = dict(zip(classes.astype(int), weights))
    print(f"[INFO] Class weights: {class_weight_dict}")
    return class_weight_dict


# =============================================================================
# 3. MODEL ARCHITECTURE
# =============================================================================

def build_model(input_dim: int) -> keras.Model:
    """
    Build a regularised feed-forward Neural Network for binary classification.

    Architecture
    ------------
    Input (input_dim)
        ↓
    [Dense(256, relu) → BatchNorm → Dropout(0.40)] ← Layer 1
        ↓
    [Dense(128, relu) → BatchNorm → Dropout(0.30)] ← Layer 2
        ↓
    [Dense(64,  relu) → BatchNorm → Dropout(0.20)] ← Layer 3
        ↓
    Dense(1, sigmoid)                              ← Output

    Regularisation choices
    ----------------------
    * L2(1e-4) on every Dense kernel — penalises large weights during training.
    * BatchNormalization — normalises layer inputs, acts as a mild regulariser
      and speeds up convergence significantly.
    * Dropout — rate decreases as we get deeper (less capacity = less overfitting
      risk in later layers).

    Parameters
    ----------
    input_dim : int
        Number of input features (determined at runtime from data).

    Returns
    -------
    keras.Model (uncompiled)
    """
    l2_reg = regularizers.L2(L2_LAMBDA)
    units: list = list(HIDDEN_UNITS)
    drops: list = list(DROPOUT_RATES)

    # Use the Keras Functional API for clarity and flexibility
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    # ── Hidden layer 1 ────────────────────────────────────────────────────────
    x = layers.Dense(
        units[0],
        activation="relu",
        kernel_regularizer=l2_reg,
        name="dense_1",
    )(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(drops[0], seed=RANDOM_SEED, name="dropout_1")(x)

    # ── Hidden layer 2 ────────────────────────────────────────────────────────
    x = layers.Dense(
        units[1],
        activation="relu",
        kernel_regularizer=l2_reg,
        name="dense_2",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(drops[1], seed=RANDOM_SEED, name="dropout_2")(x)

    # ── Hidden layer 3 ────────────────────────────────────────────────────────
    x = layers.Dense(
        units[2],
        activation="relu",
        kernel_regularizer=l2_reg,
        name="dense_3",
    )(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(drops[2], seed=RANDOM_SEED, name="dropout_3")(x)

    # ── Output layer ──────────────────────────────────────────────────────────
    # sigmoid squashes the output to [0, 1], interpreted as P(y=1 | x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="BankMarketing_NN")
    return model


# =============================================================================
# 4. COMPILATION & TRAINING
# =============================================================================

def compile_and_train(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight_dict: dict,
    checkpoint_path: Path,
) -> keras.callbacks.History:
    """
    Compile the model and run the training loop with callbacks.

    Loss & Optimizer
    ----------------
    * binary_crossentropy — the canonical loss for binary classification.
      It is numerically well-behaved with sigmoid output.
    * Adam(lr=1e-3) — adaptive moment estimation; a robust default for most
      tabular NN tasks. The LR will be halved automatically by ReduceLROnPlateau.

    Monitored Metrics (logged each epoch)
    --------------------------------------
    * AUC (ROC) — primary metric, robust to class imbalance.
    * Precision and Recall — to track the precision-recall trade-off live.

    Callbacks
    ---------
    * EarlyStopping  — monitors val_auc (higher=better); restores the best
      weights seen across all epochs so we get the optimal checkpoint for free.
    * ReduceLROnPlateau — halves the learning rate if val_loss stalls for 7
      consecutive epochs, allowing finer convergence near the optimum.
    * ModelCheckpoint — also saves the best val_auc checkpoint to disk so it
      survives even if the Python process is interrupted.

    Parameters
    ----------
    checkpoint_path : Path
        Where to save the best Keras model weights during training.

    Returns
    -------
    history : keras.callbacks.History
        Contains per-epoch loss and metric values for plotting.
    """
    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    print("\n[INFO] Model Summary:")
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=EARLY_STOP_PATIENCE,
            mode="max",                  # higher AUC is better
            restore_best_weights=True,   # automatically loads best epoch weights
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
    ]

    print(f"\n[INFO] Starting training (max {MAX_EPOCHS} epochs, "
          f"batch_size={BATCH_SIZE}, "
          f"early_stop_patience={EARLY_STOP_PATIENCE})...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,   # balances the loss per sample
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n[INFO] Training finished at epoch {len(history.epoch)}.")
    return history


# =============================================================================
# 5. EVALUATION
# =============================================================================

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate the trained model on the hold-out test set.

    WHY use the TEST set here (not val)?
    The validation set was used to:
      (a) drive EarlyStopping, and
      (b) monitor ReduceLROnPlateau.
    Therefore it has been "seen" by the training process (indirectly).
    The test set is truly held-out — its metrics give the fairest comparison
    against XGBoost and other baselines.

    Parameters
    ----------
    threshold : float, optional
        Decision threshold for converting probabilities to binary labels.
        Defaults to HYPERPARAMS["threshold"] (0.5).
        Use tune_threshold.py to find the optimal F1/recall trade-off.

    Returns
    -------
    metrics : dict  {metric_name: value}
    """
    threshold = threshold if threshold is not None else DECISION_THRESHOLD

    print(f"\n[INFO] Evaluating on Test Set (threshold={threshold:.2f})...")

    # Raw probabilities P(y=1 | x), shape (n_samples, 1)
    y_prob = model.predict(X_test, verbose=0).ravel()

    # Convert probabilities → binary labels using the chosen threshold
    y_pred = (y_prob >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
    }

    # ── Console Output ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("NEURAL NETWORK — TEST SET EVALUATION")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

    # ── WandB Logging ─────────────────────────────────────────────────────────
    log_metrics({
        "test_accuracy":  metrics["Accuracy"],
        "test_precision": metrics["Precision"],
        "test_recall":    metrics["Recall"],
        "test_f1_score":  metrics["F1-Score"],
        "test_roc_auc":   metrics["ROC-AUC"],
    })
    log_confusion_matrix(y_test, y_pred, class_names=["No", "Yes"])

    return metrics, y_pred, y_prob


# =============================================================================
# 6. PLOTTING
# =============================================================================

def plot_results(
    history: keras.callbacks.History,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    save_dir: Path,
) -> None:
    """
    Generate and save three diagnostic plots:

    1. Training curves — Loss & AUC vs. epoch (train vs. validation).
       Useful for diagnosing overfitting (gap between train and val curves).

    2. Confusion Matrix — counts of TP / TN / FP / FN on the test set.
       Use this to understand the type of errors the model makes.

    3. ROC Curve — True Positive Rate vs. False Positive Rate across all
       thresholds, with the AUC score annotated.
    """
    # ── 1. Training Curves ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Neural Network — Training History", fontsize=14, fontweight="bold")

    # Loss subplot
    axes[0].plot(history.history["loss"],     label="Train Loss",  color="#2196F3")
    axes[0].plot(history.history["val_loss"], label="Val Loss",    color="#F44336", linestyle="--")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC subplot
    axes[1].plot(history.history["auc"],     label="Train AUC", color="#4CAF50")
    axes[1].plot(history.history["val_auc"], label="Val AUC",   color="#FF9800", linestyle="--")
    axes[1].set_title("AUC per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curves_path = save_dir / "nn_training_curves.png"
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close()
    log_image(str(curves_path), "training_curves")
    print(f"[INFO] Training curves saved to {curves_path}")

    # ── 2. Confusion Matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: No", "Pred: Yes"],
        yticklabels=["True: No", "True: Yes"],
        cbar=False,
    )
    plt.title("Confusion Matrix — Neural Network (Test Set)", fontsize=13, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = save_dir / "confusion_matrix_nn.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    log_image(str(cm_path), "confusion_matrix")
    print(f"[INFO] Confusion matrix saved to {cm_path}")

    # ── 3. ROC Curve ──────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#9C27B0", linewidth=2,
             label=f"Neural Network (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.fill_between(fpr, tpr, alpha=0.08, color="#9C27B0")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve — Neural Network (Test Set)", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = save_dir / "roc_curve_nn.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    log_image(str(roc_path), "roc_curve")
    print(f"[INFO] ROC curve saved to {roc_path}")


# =============================================================================
# 7. SAVE ARTIFACTS
# =============================================================================

def save_artifacts(model: keras.Model, save_dir: Path) -> Path:
    """
    Persist the trained Keras model to disk in the native `.keras` format.

    The `.keras` format (TF ≥ 2.12) stores:
      • Model graph (architecture)
      • Trained weights
      • Optimiser state (allows resuming training)
      • Training configuration (compile settings)

    To load later:
        model = keras.models.load_model('src/models/neural_network_model.keras')

    Returns
    -------
    save_path : Path to the saved model file.
    """
    save_path = save_dir / "neural_network_model.keras"
    model.save(save_path)
    print(f"\n[INFO] Keras model saved to {save_path}")

    # Log to WandB as a model artifact for experiment tracking
    log_artifact(str(save_path), artifact_name="neural_network_keras", artifact_type="model")

    return save_path


# =============================================================================
# 8. MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # ── WandB Initialisation ──────────────────────────────────────────────────
    # Tags signal the project phase (phase-4) for filtering in the WandB dashboard
    init_wandb(
        run_name="neural-network-keras",
        tags=["neural-network", "tensorflow", "keras", "phase-4"],
    )

    # Log all hyperparameters to WandB for reproducibility
    log_config({
        "model_type": "NeuralNetwork_Keras",
        **HYPERPARAMS_LOG,
        "framework": f"TensorFlow {tf.__version__}",
        "random_seed": RANDOM_SEED,
    })

    try:
        # ── Step 1: Load Data ─────────────────────────────────────────────────
        X_train, y_train, X_val, y_val, X_test, y_test = load_data()
        input_dim = X_train.shape[1]

        # ── Step 2: Class Weights ─────────────────────────────────────────────
        class_weight_dict = compute_class_weights(y_train)

        # ── Step 3: Build Model ───────────────────────────────────────────────
        print("\n[INFO] Building Neural Network...")
        model = build_model(input_dim=input_dim)

        # ── Step 4: Train ─────────────────────────────────────────────────────
        checkpoint_path = MODELS_DIR / "neural_network_model.keras"
        history = compile_and_train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            class_weight_dict=class_weight_dict,
            checkpoint_path=checkpoint_path,
        )

        # ── Step 5: Evaluate on Test Set ──────────────────────────────────────
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

        # ── Step 6: Plot & Save Charts ────────────────────────────────────────
        plot_results(history, y_test, y_pred, y_prob, RESULTS_DIR)

        # ── Step 7: Save Model ────────────────────────────────────────────────
        save_artifacts(model, MODELS_DIR)

        print("\n[SUCCESS] Neural Network training pipeline completed successfully.")
        print(f"          Model → {MODELS_DIR / 'neural_network_model.keras'}")
        print(f"          Charts → {RESULTS_DIR}")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        raise

    finally:
        # Always close the WandB run — even if an error occurred
        finish_wandb()
