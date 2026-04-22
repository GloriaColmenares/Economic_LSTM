# 01_Python_code/deep_learning_methods/bidirectional_lstm.py

from __future__ import annotations

# Standard library imports
import copy
import os
import sys
import time
import random
import logging
import warnings
from pathlib import Path
from typing import Any

# Third-party imports
import torch
import optuna
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# --------------------------------------------------------------------------------------
# Import project paths safely
# --------------------------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
PROJECT_CODE_ROOT = CURRENT_FILE.parents[1]  # 01_Python_code
if str(PROJECT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_ROOT))

from paths import BASE_INPUT_PATH, BASE_OUTPUT_PATH  # noqa: E402

# --------------------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LOGGER.info("Using device: %s", DEVICE)

plt.style.use("default")
plt.rcParams["figure.figsize"] = (100, 10)
plt.rcParams["font.size"] = 14
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.color"] = "#cccccc"
plt.rcParams["text.color"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["axes.grid"] = True

LOOKBACKS = [42, 168]
PRODUCT_SIGNS = ["NEG", "POS"]
PRICE_TYPES = ["AVERAGE", "MARGINAL"]

GENERATOR = torch.Generator()
GENERATOR.manual_seed(SEED)


# --------------------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------------------

def seed_worker(worker_id: int) -> None:
    """
    Seed DataLoader workers for reproducibility.

    Parameters:
        worker_id: Worker identifier.

    Returns:
        None
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------------------
# Dataset wrapper
# --------------------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """
    Dataset wrapper for time series tensors.

    Parameters:
        X: Input tensor.
        y: Target tensor.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Return dataset length.

        Parameters:
            None

        Returns:
            Number of rows.
        """
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one sample.

        Parameters:
            i: Sample index.

        Returns:
            Tuple of input and target tensors.
        """
        return self.X[i], self.y[i]


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM with dropout and linear output layer.

    Parameters:
        input_size: Number of features per timestep.
        hidden_size: Hidden size.
        num_stacked_layers: Number of stacked LSTM layers.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_stacked_layers: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_stacked_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
            x: Input tensor with shape [batch, seq_len, input_size].

        Returns:
            Output tensor with shape [batch, 1].
        """
        batch_size = x.size(0)
        h0 = torch.zeros(
            self.num_stacked_layers * 2,
            batch_size,
            self.hidden_size,
            device=x.device,
        )
        c0 = torch.zeros(
            self.num_stacked_layers * 2,
            batch_size,
            self.hidden_size,
            device=x.device,
        )
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# --------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------

def preprocess_dataset(
    dataset: pd.DataFrame,
    product_sign: str,
    price_type: str,
    lookback: int,
) -> tuple[pd.DataFrame, MinMaxScaler, str, int, int]:
    """
    Prepare lagged dataset with no leakage.

    Important:
        Scaling is fit on the training split only, then applied to validation and test.

    Parameters:
        dataset: Full processed dataset.
        product_sign: Regulation direction.
        price_type: Price type.
        lookback: Number of lag observations.

    Returns:
        Processed DataFrame, fitted scaler, price column name, train limit, validation limit.
    """
    price_column = f"{product_sign}_GERMANY_{price_type}_CAPACITY_PRICE_[(EUR/MW)/h]"
    price_df = pd.DataFrame(dataset[price_column]).copy()

    price_df["log_price"] = np.log1p(price_df[price_column])

    lag_cols = [f"lag_{lag}" for lag in range(1, lookback + 1)]
    for lag, col in enumerate(lag_cols, 1):
        price_df[col] = price_df["log_price"].shift(lag)

    price_df.dropna(inplace=True)

    train_limit = int(len(price_df) * 0.70)
    validation_limit = int(len(price_df) * 0.85)

    train_df = price_df.iloc[:train_limit].copy()
    val_df = price_df.iloc[train_limit:validation_limit].copy()
    test_df = price_df.iloc[validation_limit:].copy()

    scaled_columns = ["log_price"] + lag_cols
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_df[scaled_columns] = scaler.fit_transform(train_df[scaled_columns])
    val_df[scaled_columns] = scaler.transform(val_df[scaled_columns])
    test_df[scaled_columns] = scaler.transform(test_df[scaled_columns])

    price_df = pd.concat([train_df, val_df, test_df], axis=0)

    price_df.drop(columns=[price_column], inplace=True)
    price_df = price_df.loc[:, ::-1]
    price_df = price_df.astype(np.float32)

    return price_df, scaler, price_column, train_limit, validation_limit


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Parameters:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size: Batch size.

    Returns:
        Train, validation, and test loaders.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=GENERATOR,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=GENERATOR,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=GENERATOR,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------------------
# CV helpers
# --------------------------------------------------------------------------------------

def time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_class: type[nn.Module],
    model_params: dict[str, Any],
    n_splits: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    epochs: int = 50,
    patience: int = 10,
    device_name: str = "cuda",
) -> tuple[list[float], list[nn.Module]]:
    """
    Run time-series cross-validation.

    Parameters:
        X: Input array.
        y: Target array.
        model_class: Model class.
        model_params: Model parameters.
        n_splits: Number of CV folds.
        batch_size: Batch size.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        epochs: Maximum epochs.
        patience: Early stopping patience.
        device_name: Device string.

    Returns:
        CV scores and CV-trained models.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores: list[float] = []
    cv_models: list[nn.Module] = []
    fold = 1

    for train_idx, val_idx in tscv.split(X):
        print(f"Training fold {fold}/{n_splits}")

        X_train_fold = torch.tensor(X[train_idx]).float()
        y_train_fold = torch.tensor(y[train_idx]).float()
        X_val_fold = torch.tensor(X[val_idx]).float()
        y_val_fold = torch.tensor(y[val_idx]).float()

        train_dataset = TimeSeriesDataset(X_train_fold, y_train_fold)
        val_dataset = TimeSeriesDataset(X_val_fold, y_val_fold)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=GENERATOR,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=GENERATOR,
            num_workers=0,
        )

        model = model_class(**model_params).to(device_name)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_val_loss = float("inf")
        counter = 0
        best_model_state: dict[str, torch.Tensor] | None = None

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device_name)
                y_batch = y_batch.to(device_name)

                optimizer.zero_grad()
                output = model(X_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device_name)
                    y_batch = y_batch.to(device_name)
                    output = model(X_batch)
                    loss = loss_function(output, y_batch)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)

            if epoch % 10 == 0:
                print(f"Fold {fold}, Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_model_state is None:
            raise RuntimeError(f"No best model state captured for fold {fold}.")

        model.load_state_dict(best_model_state)

        model.eval()
        final_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device_name)
                y_batch = y_batch.to(device_name)
                output = model(X_batch)
                loss = loss_function(output, y_batch)
                final_val_loss += loss.item()

        final_val_loss = final_val_loss / len(val_loader)
        cv_scores.append(final_val_loss)
        cv_models.append(model)

        print(f"Fold {fold} validation loss: {final_val_loss:.6f}")
        fold += 1

    return cv_scores, cv_models


def visualize_ts_cv_splits(
    X: np.ndarray,
    index: pd.Index,
    save_path: Path,
    n_splits: int = 5,
) -> None:
    """
    Save a visualization of time-series CV splits.

    Parameters:
        X: Input array used in CV.
        index: Datetime index corresponding to X.
        save_path: Output file path.
        n_splits: Number of CV folds.

    Returns:
        None
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    plt.figure(figsize=(15, 8))
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        fold_train_dates = index[train_idx]
        fold_test_dates = index[test_idx]

        plt.scatter(
            fold_train_dates,
            [i + 0.1] * len(fold_train_dates),
            c="blue",
            s=10,
            label="Train" if i == 0 else "",
        )
        plt.scatter(
            fold_test_dates,
            [i + 0.2] * len(fold_test_dates),
            c="red",
            s=10,
            label="Validation" if i == 0 else "",
        )

    plt.yticks(np.arange(0.15, n_splits + 0.15, 1), [f"Fold {i + 1}" for i in range(n_splits)])
    plt.title("Time Series Cross-Validation Splits")
    plt.ylabel("CV Iteration")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_ensemble(
    cv_models: list[nn.Module],
    X_test: np.ndarray,
    device_name: str = "cuda",
) -> np.ndarray:
    """
    Create average ensemble predictions.

    Parameters:
        cv_models: CV-trained models.
        X_test: Input array.
        device_name: Device string.

    Returns:
        Ensemble predictions.
    """
    X_test_tensor = torch.tensor(X_test).float().to(device_name)

    predictions = []
    for model in cv_models:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_tensor)
            predictions.append(pred.cpu().numpy())

    return np.mean(predictions, axis=0)


def create_ensemble_recency_weighted(
    cv_models: list[nn.Module],
    X_test: np.ndarray,
    device_name: str = "cuda",
    recency_factor: float = 1.5,
) -> np.ndarray:
    """
    Create recency-weighted ensemble predictions.

    Parameters:
        cv_models: CV-trained models.
        X_test: Input array.
        device_name: Device string.
        recency_factor: Weight exponent for more recent models.

    Returns:
        Weighted ensemble predictions.
    """
    X_test_tensor = torch.tensor(X_test).float().to(device_name)

    num_models = len(cv_models)
    weights = np.array([(i + 1) ** recency_factor for i in range(num_models)], dtype=np.float32)
    weights = weights / weights.sum()

    predictions = []
    for model in cv_models:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_tensor)
            predictions.append(pred.cpu().numpy())

    weighted_predictions = np.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        weighted_predictions += weights[i] * pred

    return weighted_predictions


# --------------------------------------------------------------------------------------
# Optuna training helper
# --------------------------------------------------------------------------------------

def train_and_evaluate(
    trial: optuna.trial.Trial,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    patience: int = 5,
    seed_value: int = 42,
) -> float:
    """
    Train a model inside an Optuna trial and return best validation loss.

    Parameters:
        trial: Optuna trial.
        model: Model.
        optimizer: Optimizer.
        loss_function: Loss function.
        train_loader: Training loader.
        val_loader: Validation loader.
        num_epochs: Maximum epochs.
        patience: Early stopping patience.
        seed_value: Reproducibility seed.

    Returns:
        Best validation loss.
    """
    start_time = time.time()

    best_val_loss = float("inf")
    counter = 0

    for epoch in range(num_epochs):
        torch.manual_seed(seed_value + epoch)
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()

        torch.manual_seed(seed_value)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                output = model(X_batch)
                loss = loss_function(output, y_batch)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    execution_time = time.time() - start_time
    print(f"Training completed in {execution_time:.2f} seconds with val_loss: {best_val_loss:.6f}")

    return best_val_loss


# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------

def determine_product(hour: int) -> str:
    """
    Map hour to 4-hour product block.

    Parameters:
        hour: Hour from timestamp.

    Returns:
        Product label.
    """
    if hour == 0:
        return "00_04"
    if hour == 4:
        return "04_08"
    if hour == 8:
        return "08_12"
    if hour == 12:
        return "12_16"
    if hour == 16:
        return "16_20"
    if hour == 20:
        return "20_24"
    return "all"


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point.

    Parameters:
        None

    Returns:
        None
    """
    output_folder = BASE_OUTPUT_PATH / "bidirectional_lstm_model_output"
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_path = BASE_INPUT_PATH / "processed_afrr_data.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    try:
        processed_afrr_data = pd.read_csv(dataset_path, parse_dates=["DATE"], index_col=["DATE"])
    except Exception as exc:
        raise RuntimeError(f"Error reading dataset: {exc}") from exc

    for lookback in LOOKBACKS:
        all_models_output: list[dict[str, Any]] = []

        for product_sign in PRODUCT_SIGNS:
            for price_type in PRICE_TYPES:
                total_start_time = time.time()

                price_df, scaler, price_column, train_limit, validation_limit = preprocess_dataset(
                    dataset=processed_afrr_data,
                    product_sign=product_sign,
                    price_type=price_type,
                    lookback=lookback,
                )

                output_path = output_folder
                output_path.mkdir(parents=True, exist_ok=True)

                price_df_init = pd.DataFrame(processed_afrr_data[price_column]).loc[price_df.index].copy()
                price_array = price_df.to_numpy()

                X = price_array[:, :-1]
                y = price_array[:, -1]

                X_cv_init = X[:validation_limit]
                y_cv_init = y[:validation_limit]

                X_train = X[:train_limit]
                y_train = y[:train_limit]
                X_val = X[train_limit:validation_limit]
                y_val = y[train_limit:validation_limit]
                X_test = X[validation_limit:]
                y_test = y[validation_limit:]

                X_cv = X_cv_init.reshape(-1, lookback, 1)
                y_cv = y_cv_init.reshape(-1, 1)

                X_train = X_train.reshape(-1, lookback, 1)
                X_val = X_val.reshape(-1, lookback, 1)
                X_test = X_test.reshape(-1, lookback, 1)

                y_train = y_train.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)

                X_train_t = torch.tensor(X_train).float()
                X_val_t = torch.tensor(X_val).float()
                X_test_t = torch.tensor(X_test).float()

                y_train_t = torch.tensor(y_train).float()
                y_val_t = torch.tensor(y_val).float()
                y_test_t = torch.tensor(y_test).float()

                train_dataset = TimeSeriesDataset(X_train_t, y_train_t)
                val_dataset = TimeSeriesDataset(X_val_t, y_val_t)
                test_dataset = TimeSeriesDataset(X_test_t, y_test_t)

                def create_dataloaders_local(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
                    """
                    Create local dataloaders for this run.

                    Parameters:
                        batch_size: Batch size.

                    Returns:
                        Train, validation, and test loaders.
                    """
                    return create_dataloaders(
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        test_dataset=test_dataset,
                        batch_size=batch_size,
                    )

                phase1_best_params: dict[str, Any] = {}
                phase2_best_params: dict[str, Any] = {}
                phase3_best_params: dict[str, Any] = {}

                def phase1_objective(trial: optuna.trial.Trial) -> float:
                    """
                    Optimize architecture hyperparameters.

                    Parameters:
                        trial: Optuna trial.

                    Returns:
                        Validation loss.
                    """
                    hidden_size = trial.suggest_int("hidden_size", 64, 128)
                    num_stacked_layers = trial.suggest_int("num_stacked_layers", 1, 2)

                    learning_rate = 0.001
                    dropout_rate = 0.2
                    batch_size = 64

                    print(f"Phase 1 - Trial #{trial.number}")

                    model = BiLSTM(
                        input_size=1,
                        hidden_size=hidden_size,
                        num_stacked_layers=num_stacked_layers,
                        dropout_rate=dropout_rate,
                    ).to(DEVICE)

                    train_loader, val_loader, _ = create_dataloaders_local(batch_size)
                    loss_function = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    return train_and_evaluate(
                        trial=trial,
                        model=model,
                        optimizer=optimizer,
                        loss_function=loss_function,
                        train_loader=train_loader,
                        val_loader=val_loader,
                    )

                def phase2_objective(trial: optuna.trial.Trial) -> float:
                    """
                    Optimize regularization hyperparameters.

                    Parameters:
                        trial: Optuna trial.

                    Returns:
                        Validation loss.
                    """
                    hidden_size = phase1_best_params["hidden_size"]
                    num_stacked_layers = phase1_best_params["num_stacked_layers"]

                    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)
                    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

                    learning_rate = 0.001
                    batch_size = 64

                    print(f"Phase 2 - Trial #{trial.number}")

                    model = BiLSTM(
                        input_size=1,
                        hidden_size=hidden_size,
                        num_stacked_layers=num_stacked_layers,
                        dropout_rate=dropout_rate,
                    ).to(DEVICE)

                    train_loader, val_loader, _ = create_dataloaders_local(batch_size)
                    loss_function = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                    return train_and_evaluate(
                        trial=trial,
                        model=model,
                        optimizer=optimizer,
                        loss_function=loss_function,
                        train_loader=train_loader,
                        val_loader=val_loader,
                    )

                def phase3_objective(trial: optuna.trial.Trial) -> float:
                    """
                    Optimize learning hyperparameters.

                    Parameters:
                        trial: Optuna trial.

                    Returns:
                        Validation loss.
                    """
                    hidden_size = phase1_best_params["hidden_size"]
                    num_stacked_layers = phase1_best_params["num_stacked_layers"]

                    dropout_rate = phase2_best_params["dropout_rate"]
                    weight_decay = phase2_best_params["weight_decay"]

                    learning_rate = trial.suggest_float("learning_rate", 5e-4, 1e-3, log=True)
                    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

                    print(f"Phase 3 - Trial #{trial.number}")

                    model = BiLSTM(
                        input_size=1,
                        hidden_size=hidden_size,
                        num_stacked_layers=num_stacked_layers,
                        dropout_rate=dropout_rate,
                    ).to(DEVICE)

                    train_loader, val_loader, _ = create_dataloaders_local(batch_size)
                    loss_function = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                    return train_and_evaluate(
                        trial=trial,
                        model=model,
                        optimizer=optimizer,
                        loss_function=loss_function,
                        train_loader=train_loader,
                        val_loader=val_loader,
                    )

                def optimize_lstm_hyperparameters(n_trials_per_phase: int = 20) -> dict[str, Any]:
                    """
                    Run the 3-phase Optuna optimization.

                    Parameters:
                        n_trials_per_phase: Number of trials per phase.

                    Returns:
                        Combined best parameters.
                    """
                    nonlocal phase1_best_params, phase2_best_params, phase3_best_params

                    print("Starting Phase 1: Architecture Optimization")
                    phase1_study = optuna.create_study(
                        direction="minimize",
                        sampler=TPESampler(seed=42),
                        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
                        study_name="phase1_architecture",
                    )
                    phase1_study.optimize(phase1_objective, n_trials=n_trials_per_phase)
                    phase1_best_params = phase1_study.best_params

                    print("\nStarting Phase 2: Regularization Optimization")
                    phase2_study = optuna.create_study(
                        direction="minimize",
                        sampler=TPESampler(seed=42),
                        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
                        study_name="phase2_regularization",
                    )
                    phase2_study.optimize(phase2_objective, n_trials=n_trials_per_phase)
                    phase2_best_params = phase2_study.best_params

                    print("\nStarting Phase 3: Learning Parameters Optimization")
                    phase3_study = optuna.create_study(
                        direction="minimize",
                        sampler=TPESampler(seed=42),
                        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
                        study_name="phase3_learning",
                    )
                    phase3_study.optimize(phase3_objective, n_trials=n_trials_per_phase)
                    phase3_best_params = phase3_study.best_params

                    return {
                        **phase1_best_params,
                        **phase2_best_params,
                        **phase3_best_params,
                    }

                best_params = optimize_lstm_hyperparameters(n_trials_per_phase=20)

                model_params = {
                    "input_size": 1,
                    "hidden_size": best_params["hidden_size"],
                    "num_stacked_layers": best_params["num_stacked_layers"],
                    "dropout_rate": best_params["dropout_rate"],
                }

                visualize_ts_cv_splits(
                    X=X_cv,
                    index=price_df_init.index[:validation_limit],
                    save_path=output_path / f"bidirectional_lstm_ts{lookback}_{product_sign}_{price_type}_cv_splits.png",
                    n_splits=5,
                )

                cv_scores, cv_models = time_series_cv(
                    X=X_cv,
                    y=y_cv,
                    model_class=BiLSTM,
                    model_params=model_params,
                    n_splits=5,
                    batch_size=best_params.get("batch_size", 64),
                    learning_rate=best_params.get("learning_rate", 0.001),
                    weight_decay=best_params.get("weight_decay", 0.0),
                    epochs=100,
                    patience=15,
                    device_name=DEVICE,
                )

                ensemble_val_preds = create_ensemble(cv_models, X_val, device_name=DEVICE)
                recency_weighted_val_preds = create_ensemble_recency_weighted(
                    cv_models,
                    X_val,
                    device_name=DEVICE,
                    recency_factor=1.5,
                )

                ensemble_test_preds = create_ensemble(cv_models, X_test, device_name=DEVICE)
                recency_weighted_test_preds = create_ensemble_recency_weighted(
                    cv_models,
                    X_test,
                    device_name=DEVICE,
                    recency_factor=1.5,
                )

                y_val_np = y_val_t.cpu().numpy()
                dummy_val = np.zeros((len(y_val_np), lookback + 1), dtype=np.float32)
                dummy_val[:, -1] = y_val_np.flatten()

                dummy_pred = np.zeros((len(ensemble_val_preds), lookback + 1), dtype=np.float32)
                dummy_pred[:, -1] = ensemble_val_preds.flatten()

                dummy_recency_pred = np.zeros((len(recency_weighted_val_preds), lookback + 1), dtype=np.float32)
                dummy_recency_pred[:, -1] = recency_weighted_val_preds.flatten()

                y_val_inv = scaler.inverse_transform(dummy_val)[:, -1]
                pred_inv = scaler.inverse_transform(dummy_pred)[:, -1]
                recency_pred_inv = scaler.inverse_transform(dummy_recency_pred)[:, -1]

                y_val_original = np.expm1(y_val_inv)
                ensemble_val_pred_original = np.expm1(pred_inv)
                recency_val_pred_original = np.expm1(recency_pred_inv)

                y_test_np = y_test_t.cpu().numpy()
                dummy_test = np.zeros((len(y_test_np), lookback + 1), dtype=np.float32)
                dummy_test[:, -1] = y_test_np.flatten()

                dummy_pred = np.zeros((len(ensemble_test_preds), lookback + 1), dtype=np.float32)
                dummy_pred[:, -1] = ensemble_test_preds.flatten()

                dummy_recency_pred = np.zeros((len(recency_weighted_test_preds), lookback + 1), dtype=np.float32)
                dummy_recency_pred[:, -1] = recency_weighted_test_preds.flatten()

                y_test_inv = scaler.inverse_transform(dummy_test)[:, -1]
                pred_inv = scaler.inverse_transform(dummy_pred)[:, -1]
                recency_pred_inv = scaler.inverse_transform(dummy_recency_pred)[:, -1]

                y_test_original = np.expm1(y_test_inv)
                ensemble_test_pred_original = np.expm1(pred_inv)
                recency_test_pred_original = np.expm1(recency_pred_inv)

                best_model = BiLSTM(**model_params).to(DEVICE)
                loss_function = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    best_model.parameters(),
                    lr=best_params.get("learning_rate", 0.001),
                    weight_decay=best_params.get("weight_decay", 0.0),
                )

                train_loader, val_loader, test_loader = create_dataloaders_local(
                    batch_size=best_params.get("batch_size", 64)
                )

                best_val_loss = float("inf")
                epochs = 100
                patience = 15
                counter = 0
                train_losses: list[float] = []
                val_losses: list[float] = []
                best_model_state: dict[str, torch.Tensor] | None = None

                for epoch in range(epochs):
                    best_model.train()
                    running_loss = 0.0
                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(DEVICE)
                        y_batch = y_batch.to(DEVICE)

                        optimizer.zero_grad()
                        output = best_model(X_batch)
                        loss = loss_function(output, y_batch)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                    train_loss = running_loss / len(train_loader)
                    train_losses.append(train_loss)

                    best_model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch = X_batch.to(DEVICE)
                            y_batch = y_batch.to(DEVICE)
                            output = best_model(X_batch)
                            loss = loss_function(output, y_batch)
                            val_loss += loss.item()

                    val_loss = val_loss / len(val_loader)
                    val_losses.append(val_loss)

                    if epoch % 10 == 0:
                        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.6f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        counter = 0
                        best_model_state = copy.deepcopy(best_model.state_dict())
                    else:
                        counter += 1
                        if counter >= patience:
                            print(f"Early stopping at epoch {epoch + 1}")
                            break

                if best_model_state is None:
                    raise RuntimeError("No best final model state captured.")

                best_model.load_state_dict(best_model_state)

                plt.figure(figsize=(12, 6))
                plt.plot(train_losses, label="Training Loss")
                plt.plot(val_losses, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss (MSE)")
                plt.title(f"Training and Validation Loss for {product_sign} {price_type}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                best_epoch = int(np.argmin(val_losses))
                plt.scatter(
                    best_epoch,
                    val_losses[best_epoch],
                    marker="o",
                    color="red",
                    s=100,
                    label=f"Best Model (Epoch {best_epoch + 1})",
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    output_path / f"bidirectional_lstm_ts{lookback}_{product_sign}_{price_type}_training_validation_loss.png"
                )
                plt.close()

                best_model.eval()
                single_val_preds = []
                with torch.no_grad():
                    for X_batch, _ in val_loader:
                        X_batch = X_batch.to(DEVICE)
                        outputs = best_model(X_batch)
                        single_val_preds.extend(outputs.cpu().numpy())
                single_val_preds = np.array(single_val_preds)

                best_model.eval()
                single_test_preds = []
                with torch.no_grad():
                    for X_batch, _ in test_loader:
                        X_batch = X_batch.to(DEVICE)
                        outputs = best_model(X_batch)
                        single_test_preds.extend(outputs.cpu().numpy())
                single_test_preds = np.array(single_test_preds)

                dummy_pred = np.zeros((len(single_val_preds), lookback + 1), dtype=np.float32)
                dummy_pred[:, -1] = single_val_preds.flatten()
                pred_inv = scaler.inverse_transform(dummy_pred)[:, -1]
                single_val_pred_original = np.expm1(pred_inv)

                dummy_pred = np.zeros((len(single_test_preds), lookback + 1), dtype=np.float32)
                dummy_pred[:, -1] = single_test_preds.flatten()
                pred_inv = scaler.inverse_transform(dummy_pred)[:, -1]
                single_test_pred_original = np.expm1(pred_inv)

                total_execution_time = time.time() - total_start_time

                df_val = pd.DataFrame(
                    {
                        "ACTUAL_VALUE": y_val_original,
                        "D+1_ENSEMBLE_MODEL": ensemble_val_pred_original,
                        "D+1_RECENCY_WEIGHTED": recency_val_pred_original,
                        "D+1_SINGLE_MODEL": single_val_pred_original,
                    }
                )

                df_test = pd.DataFrame(
                    {
                        "ACTUAL_VALUE": y_test_original,
                        "D+1_ENSEMBLE_MODEL": ensemble_test_pred_original,
                        "D+1_RECENCY_WEIGHTED": recency_test_pred_original,
                        "D+1_SINGLE_MODEL": single_test_pred_original,
                    }
                )

                val_indices = price_df_init[train_limit:validation_limit].index
                test_indices = price_df_init[validation_limit:].index

                df_val.index = pd.DatetimeIndex(val_indices)
                df_test.index = pd.DatetimeIndex(test_indices)

                df_val["PRODUCT"] = df_val.index.hour.map(determine_product)
                df_test["PRODUCT"] = df_test.index.hour.map(determine_product)

                products = ["00_04", "04_08", "08_12", "12_16", "16_20", "20_24"]

                dfs_val = {product: df_val[df_val["PRODUCT"] == product] for product in products}
                dfs_test = {product: df_test[df_test["PRODUCT"] == product] for product in products}
                dfs_combined = {product: pd.concat([dfs_val[product], dfs_test[product]]) for product in products}

                for product in products:
                    dfs_combined[product] = dfs_combined[product].drop(columns=["PRODUCT"])
                    file_name = (
                        f"bidirectional_lstm_ts{lookback}_model_results_"
                        f"{product}_{product_sign}_{price_type}.csv"
                    )
                    file_path = output_path / file_name
                    dfs_combined[product].to_csv(file_path)

                new_row = {
                    "PRODUCT": "all",
                    "PRODUCT_SIGN": product_sign,
                    "PRICE_TYPE": price_type,
                    "TIMESTEP": lookback,
                    "HIDDEN_SIZE": best_params["hidden_size"],
                    "NUM_STACKED_LAYERS": best_params["num_stacked_layers"],
                    "LEARNING_RATE": best_params["learning_rate"],
                    "DROPOUT_RATE": best_params["dropout_rate"],
                    "WEIGHT_DECAY": best_params["weight_decay"],
                    "BATCH_SIZE": best_params["batch_size"],
                    "EXECUTION_TIME_[S]": total_execution_time,
                    "CV_MEAN_LOSS": float(np.mean(cv_scores)),
                }
                all_models_output.append(new_row)

        all_models_output_df = pd.DataFrame(all_models_output)
        output_file_path = output_path / f"bidirectional_lstm_ts{lookback}_model_overview.csv"
        all_models_output_df.to_csv(output_file_path, index=False)

    LOGGER.info("Bidirectional LSTM runs completed successfully.")


# Example usage
# python3 -m deep_learning_methods.bidirectional_lstm
if __name__ == "__main__":
    main()