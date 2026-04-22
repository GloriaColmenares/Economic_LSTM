# 01_Python_code/deep_learning_methods/lstm.py

# Standard library imports
import copy
import os
import sys
import time
import random
import logging
import warnings
from pathlib import Path

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

# Local imports
CURRENT_FILE = Path(__file__).resolve()
PROJECT_CODE_ROOT = CURRENT_FILE.parents[1]  # 01_Python_code
if str(PROJECT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_ROOT))

from paths import BASE_INPUT_PATH, BASE_OUTPUT_PATH  # noqa: E402

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reproducibility
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)

# Matplotlib parameters
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


def seed_worker(worker_id: int) -> None:
    """
    Seed worker processes for reproducibility.

    Parameters:
        worker_id: Worker identifier.

    Returns:
        None
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)


class TimeSeriesDataset(Dataset):
    """
    Dataset wrapper for time series tensors.

    Parameters:
        X: Input features tensor.
        y: Target tensor.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


class LSTM(nn.Module):
    """
    LSTM model with dropout and fully connected output layer.

    Parameters:
        input_size: Number of input features.
        hidden_size: Hidden size of the LSTM.
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
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
            x: Input tensor of shape [batch, seq_len, input_size].

        Returns:
            Output tensor of shape [batch, 1].
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def preprocess_dataset_no_leakage(
    dataset: pd.DataFrame,
    product_sign: str,
    price_type: str,
    lookback: int,
) -> tuple[pd.DataFrame, str]:
    """
    Build log-transformed lagged dataset without scaling.

    Parameters:
        dataset: Full input dataset.
        product_sign: Regulation direction.
        price_type: Price type.
        lookback: Number of lagged steps.

    Returns:
        Tuple of:
        - lagged DataFrame with columns [lag_lookback ... lag_1, log_price]
        - original price column name
    """
    price_column = f"{product_sign}_GERMANY_{price_type}_CAPACITY_PRICE_[(EUR/MW)/h]"
    price_df = pd.DataFrame(dataset[price_column]).copy()

    # log1p handles zeros safely
    price_df["log_price"] = np.log1p(price_df[price_column])

    lag_cols = [f"lag_{lag}" for lag in range(1, lookback + 1)]
    for lag, col in enumerate(lag_cols, 1):
        price_df[col] = price_df["log_price"].shift(lag)

    price_df.drop(columns=[price_column], inplace=True)
    price_df = price_df.loc[:, ::-1]
    price_df.dropna(inplace=True)
    price_df = price_df.astype(np.float32)

    return price_df, price_column


def split_train_valid_test(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Split array into train, validation, and test.

    Parameters:
        arr: Full array.

    Returns:
        train, valid, test, train_limit, validation_limit
    """
    train_limit = int(len(arr) * 0.7)
    validation_limit = int(len(arr) * 0.85)

    train = arr[:train_limit]
    valid = arr[train_limit:validation_limit]
    test = arr[validation_limit:]

    return train, valid, test, train_limit, validation_limit


def make_xy(
    scaled_array: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split scaled tabular lag data into X and y.

    Parameters:
        scaled_array: Scaled array with lag columns and target in last column.
        lookback: Number of lagged steps.

    Returns:
        X, y
    """
    X = scaled_array[:, :-1]
    y = scaled_array[:, -1]

    X = X.reshape(-1, lookback, 1)
    y = y.reshape(-1, 1)

    return X.astype(np.float32), y.astype(np.float32)


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
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


def inverse_target_only(
    scaler: MinMaxScaler,
    preds_scaled: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """
    Inverse-transform the target column only, then apply expm1.

    Parameters:
        scaler: Fitted MinMaxScaler.
        preds_scaled: Scaled predictions or targets.
        lookback: Number of lag features.

    Returns:
        Values on the original scale.
    """
    dummy = np.zeros((len(preds_scaled), lookback + 1), dtype=np.float32)
    dummy[:, -1] = preds_scaled.flatten()
    inv = scaler.inverse_transform(dummy)[:, -1]
    return np.expm1(inv)


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
    Train model and return best validation loss.

    Parameters:
        trial: Optuna trial.
        model: Model to train.
        optimizer: Optimizer.
        loss_function: Loss function.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        num_epochs: Max epochs.
        patience: Early stopping patience.
        seed_value: Base seed.

    Returns:
        Best validation loss.
    """
    start_time = time.time()
    best_val_loss = float("inf")
    counter = 0

    for epoch in range(num_epochs):
        torch.manual_seed(seed_value + epoch)
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        torch.manual_seed(seed_value)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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


def time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_class: type[nn.Module],
    model_params: dict,
    n_splits: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    epochs: int = 50,
    patience: int = 10,
    device_name: str = "cuda",
) -> tuple[list[float], list[nn.Module]]:
    """
    Time-series cross-validation.

    Parameters:
        X: Input array.
        y: Target array.
        model_class: Model class.
        model_params: Model parameters.
        n_splits: Number of CV splits.
        batch_size: Batch size.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        epochs: Max epochs.
        patience: Early stopping patience.
        device_name: Device string.

    Returns:
        cv_scores, cv_models
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores: list[float] = []
    cv_models: list[nn.Module] = []
    fold = 1

    for train_idx, val_idx in tscv.split(X):
        print(f"Training fold {fold}/{n_splits}")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        X_train_fold_t = torch.tensor(X_train_fold).float()
        y_train_fold_t = torch.tensor(y_train_fold).float()
        X_val_fold_t = torch.tensor(X_val_fold).float()
        y_val_fold_t = torch.tensor(y_val_fold).float()

        train_dataset = TimeSeriesDataset(X_train_fold_t, y_train_fold_t)
        val_dataset = TimeSeriesDataset(X_val_fold_t, y_val_fold_t)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
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
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device_name), y_batch.to(device_name)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device_name), y_batch.to(device_name)
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
            raise RuntimeError(f"No best model state captured for fold {fold}")

        model.load_state_dict(best_model_state)

        model.eval()
        final_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device_name), y_batch.to(device_name)
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
    Visualize time-series CV splits and save figure.

    Parameters:
        X: Input data.
        index: Datetime index.
        save_path: Path to save figure.
        n_splits: Number of splits.

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
        cv_models: Trained CV models.
        X_test: Test inputs.
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

    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred


def create_ensemble_recency_weighted(
    cv_models: list[nn.Module],
    X_test: np.ndarray,
    device_name: str = "cuda",
    recency_factor: float = 1.5,
) -> np.ndarray:
    """
    Create recency-weighted ensemble predictions.

    Parameters:
        cv_models: Trained CV models.
        X_test: Test inputs.
        device_name: Device string.
        recency_factor: Weighting exponent for more recent folds.

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


def determine_product(hour: int) -> str:
    """
    Map hour to product block.

    Parameters:
        hour: Hour of timestamp.

    Returns:
        Product name.
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


def run_lstm_for_configuration(
    dataset: pd.DataFrame,
    product_sign: str,
    price_type: str,
    lookback: int,
    output_path: Path,
) -> dict:
    """
    Run the full LSTM pipeline for one sign/price_type/lookback configuration.

    Parameters:
        dataset: Input dataset.
        product_sign: Regulation direction.
        price_type: Price type.
        lookback: Number of lagged steps.
        output_path: Output directory.

    Returns:
        Overview row with model settings and execution time.
    """
    total_start_time = time.time()

    price_df, price_column = preprocess_dataset_no_leakage(
        dataset=dataset,
        product_sign=product_sign,
        price_type=price_type,
        lookback=lookback,
    )

    price_df_init = pd.DataFrame(dataset[price_column]).loc[price_df.index].copy()

    full_array = price_df.to_numpy(dtype=np.float32)

    train_raw, valid_raw, test_raw, train_limit, validation_limit = split_train_valid_test(full_array)

    # IMPORTANT: fit scaler on train only to avoid leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)
    valid_scaled = scaler.transform(valid_raw)
    test_scaled = scaler.transform(test_raw)

    # CV uses train + validation only, transformed with TRAIN-fitted scaler
    cv_scaled = np.vstack([train_scaled, valid_scaled])

    X_train, y_train = make_xy(train_scaled, lookback)
    X_val, y_val = make_xy(valid_scaled, lookback)
    X_test, y_test = make_xy(test_scaled, lookback)
    X_cv, y_cv = make_xy(cv_scaled, lookback)

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
        return create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
        )

    phase1_best_params: dict = {}
    phase2_best_params: dict = {}
    phase3_best_params: dict = {}

    def phase1_objective(trial: optuna.trial.Trial) -> float:
        hidden_size = trial.suggest_int("hidden_size", 64, 128)
        num_stacked_layers = trial.suggest_int("num_stacked_layers", 1, 2)

        learning_rate = 0.001
        dropout_rate = 0.2
        batch_size = 64

        print(f"Phase 1 - Trial #{trial.number}")

        model = LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_stacked_layers=num_stacked_layers,
            dropout_rate=dropout_rate,
        ).to(device)

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
        hidden_size = phase1_best_params["hidden_size"]
        num_stacked_layers = phase1_best_params["num_stacked_layers"]

        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

        learning_rate = 0.001
        batch_size = 64

        print(f"Phase 2 - Trial #{trial.number}")

        model = LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_stacked_layers=num_stacked_layers,
            dropout_rate=dropout_rate,
        ).to(device)

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
        hidden_size = phase1_best_params["hidden_size"]
        num_stacked_layers = phase1_best_params["num_stacked_layers"]

        dropout_rate = phase2_best_params["dropout_rate"]
        weight_decay = phase2_best_params["weight_decay"]

        learning_rate = trial.suggest_float("learning_rate", 5e-4, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        print(f"Phase 3 - Trial #{trial.number}")

        model = LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_stacked_layers=num_stacked_layers,
            dropout_rate=dropout_rate,
        ).to(device)

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

    def optimize_lstm_hyperparameters(n_trials_per_phase: int = 20) -> dict:
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

        best_params = {
            **phase1_best_params,
            **phase2_best_params,
            **phase3_best_params,
        }
        return best_params

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
        save_path=output_path / f"lstm_ts{lookback}_{product_sign}_{price_type}_cv_splits.png",
        n_splits=5,
    )

    cv_scores, cv_models = time_series_cv(
        X=X_cv,
        y=y_cv,
        model_class=LSTM,
        model_params=model_params,
        n_splits=5,
        batch_size=best_params.get("batch_size", 64),
        learning_rate=best_params.get("learning_rate", 0.001),
        weight_decay=best_params.get("weight_decay", 0.0),
        epochs=100,
        patience=15,
        device_name=device,
    )

    ensemble_val_preds = create_ensemble(cv_models, X_val, device_name=device)
    recency_weighted_val_preds = create_ensemble_recency_weighted(
        cv_models,
        X_val,
        device_name=device,
        recency_factor=1.5,
    )

    ensemble_test_preds = create_ensemble(cv_models, X_test, device_name=device)
    recency_weighted_test_preds = create_ensemble_recency_weighted(
        cv_models,
        X_test,
        device_name=device,
        recency_factor=1.5,
    )

    y_val_original = inverse_target_only(scaler, y_val, lookback)
    ensemble_val_pred_original = inverse_target_only(scaler, ensemble_val_preds, lookback)
    recency_val_pred_original = inverse_target_only(scaler, recency_weighted_val_preds, lookback)

    y_test_original = inverse_target_only(scaler, y_test, lookback)
    ensemble_test_pred_original = inverse_target_only(scaler, ensemble_test_preds, lookback)
    recency_test_pred_original = inverse_target_only(scaler, recency_weighted_test_preds, lookback)

    best_model = LSTM(**model_params).to(device)
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
    train_losses = []
    val_losses = []
    best_model_state = None

    for epoch in range(epochs):
        best_model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
    plt.grid(True)

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
    plt.savefig(output_path / f"ts{lookback}_{product_sign}_{price_type}_training_validation_loss.png")
    plt.close()

    best_model.eval()
    single_val_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            outputs = best_model(X_batch)
            single_val_preds.extend(outputs.cpu().numpy())
    single_val_preds = np.array(single_val_preds)

    best_model.eval()
    single_test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = best_model(X_batch)
            single_test_preds.extend(outputs.cpu().numpy())
    single_test_preds = np.array(single_test_preds)

    single_val_pred_original = inverse_target_only(scaler, single_val_preds, lookback)
    single_pred_test_original = inverse_target_only(scaler, single_test_preds, lookback)

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
            "D+1_SINGLE_MODEL": single_pred_test_original,
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
        file_name = f"lstm_ts{lookback}_model_results_{product}_{product_sign}_{price_type}.csv"
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

    return new_row


def main() -> None:
    """
    Main entry point.

    Parameters:
        None

    Returns:
        None
    """
    output_folder = BASE_OUTPUT_PATH / "lstm_model_output"
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_path = BASE_INPUT_PATH / "processed_afrr_data.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    processed_afrr_data = pd.read_csv(dataset_path, parse_dates=["DATE"], index_col=["DATE"])

    for lookback in LOOKBACKS:
        all_models_output = []

        for product_sign in PRODUCT_SIGNS:
            for price_type in PRICE_TYPES:
                row = run_lstm_for_configuration(
                    dataset=processed_afrr_data,
                    product_sign=product_sign,
                    price_type=price_type,
                    lookback=lookback,
                    output_path=output_folder,
                )
                all_models_output.append(row)

        all_models_output_df = pd.DataFrame(all_models_output)
        output_file_path = output_folder / f"lstm_ts{lookback}_model_overview.csv"
        all_models_output_df.to_csv(output_file_path, index=False)


# Example usage
# python3 -m deep_learning_methods.lstm
if __name__ == "__main__":
    main()