# 01_Python_code/statistical_methods/arma.py

"""
ARMA forecasting script for aFRR capacity prices.

This script:
- loads the processed dataset
- fits ARMA models for each product / sign / price type combination
- performs walk-forward one-step-ahead forecasting
- saves raw prediction results per combination
- saves one combined overview file after all runs finish

Notes:
- ARMA is implemented through statsmodels' SARIMAX API with order=(p, 0, q)
- preprocessing parameters are fit on the training data only
- validation forecasts are generated after model order selection by AIC on the training set
"""

from __future__ import annotations

# Standard library imports
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Third-party imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------------------------------------------------------------------
# Import project paths safely
# --------------------------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
PROJECT_CODE_ROOT = CURRENT_FILE.parents[1]  # 01_Python_code
if str(PROJECT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_ROOT))

from paths import BASE_INPUT_PATH, BASE_OUTPUT_PATH  # noqa: E402

# --------------------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging once for the script.

    Parameters:
        level: Logging level to use.

    Returns:
        None
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------

@dataclass
class ArmaRunResult:
    """
    Stores the output metadata for one ARMA run.

    Parameters:
        product: Time product identifier.
        product_sign: Regulation direction.
        price_type: Price type.
        order: Best ARMA order as a string.
        aic: Best AIC value.
        output_file: Path to the saved raw prediction CSV.
        success: Whether the run completed successfully.
        message: Informational or error message.
    """

    product: str
    product_sign: str
    price_type: str
    order: str
    aic: Optional[float]
    output_file: str
    success: bool
    message: str


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def ensure_daily_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the DataFrame has a DatetimeIndex sorted in ascending order.

    Parameters:
        df: Input DataFrame.

    Returns:
        A sorted DataFrame with DatetimeIndex.

    Raises:
        TypeError: If the index is not a DatetimeIndex.
        ValueError: If duplicate timestamps are found.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The input DataFrame must use a pandas.DatetimeIndex.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in the index.")

    return df.sort_index()


def build_target_column(product_sign: str, price_type: str) -> str:
    """
    Build the target column name used in the dataset.

    Parameters:
        product_sign: Regulation direction, e.g. 'POS' or 'NEG'.
        price_type: Price type, e.g. 'AVERAGE' or 'MARGINAL'.

    Returns:
        The target column name.
    """
    return f"{product_sign}_GERMANY_{price_type}_CAPACITY_PRICE_[(EUR/MW)/h]"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load the processed dataset from CSV.

    Parameters:
        dataset_path: Path to the processed CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path, parse_dates=["DATE"], index_col="DATE")
    return ensure_daily_datetime_index(df)


def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time series DataFrame into train, validation, and test sets.

    Parameters:
        df: Time series DataFrame.
        train_ratio: Fraction used for training.
        valid_ratio: Fraction used for validation.

    Returns:
        Tuple of (train, validation, test).

    Raises:
        ValueError: If the resulting splits are empty or invalid.
    """
    n_obs = len(df)
    if n_obs < 10:
        raise ValueError("Not enough observations for train/validation/test split.")

    train_end = int(n_obs * train_ratio)
    valid_end = int(n_obs * (train_ratio + valid_ratio))

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError(
            f"Invalid split sizes: train={len(train)}, valid={len(valid)}, test={len(test)}"
        )

    return train, valid, test


def apply_preprocessing_from_train(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    use_capping: bool = True,
    use_log_transform: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Apply preprocessing using parameters estimated from the training set only.

    Parameters:
        train: Training DataFrame.
        valid: Validation DataFrame.
        test: Test DataFrame.
        use_capping: Whether to cap values at the train 95th percentile.
        use_log_transform: Whether to apply a natural log transform.

    Returns:
        Tuple of transformed (train, valid, test, preprocessing_info).

    Raises:
        ValueError: If log transform is requested but non-positive values exist after capping.
    """
    train_t = train.copy()
    valid_t = valid.copy()
    test_t = test.copy()

    preprocessing_info: dict[str, Any] = {
        "use_capping": use_capping,
        "use_log_transform": use_log_transform,
        "cap_value": None,
    }

    if use_capping:
        cap_value = float(train_t.quantile(0.95).iloc[0])
        preprocessing_info["cap_value"] = cap_value

        train_t = train_t.clip(upper=cap_value, axis=1)
        valid_t = valid_t.clip(upper=cap_value, axis=1)
        test_t = test_t.clip(upper=cap_value, axis=1)

    if use_log_transform:
        for split_name, split_df in [("train", train_t), ("valid", valid_t), ("test", test_t)]:
            if (split_df.iloc[:, 0] <= 0).any():
                min_value = float(split_df.iloc[:, 0].min())
                raise ValueError(
                    f"Log transform requested, but {split_name} contains non-positive values. "
                    f"Minimum value found: {min_value}"
                )

        train_t = np.log(train_t)
        valid_t = np.log(valid_t)
        test_t = np.log(test_t)

    return train_t, valid_t, test_t, preprocessing_info


def select_best_arma_order(
    train_series: pd.Series,
    max_p: int = 4,
    max_q: int = 4,
) -> tuple[tuple[int, int, int], float]:
    """
    Select the best ARMA(p, q) order using training-set AIC.

    Parameters:
        train_series: Training time series.
        max_p: Maximum AR order to test.
        max_q: Maximum MA order to test.

    Returns:
        Tuple of (best_order, best_aic).

    Raises:
        RuntimeError: If no model fit succeeds.
    """
    best_aic = np.inf
    best_order: Optional[tuple[int, int, int]] = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            order = (p, 0, q)
            try:
                model = SARIMAX(
                    endog=train_series,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)
                if fitted.aic < best_aic:
                    best_aic = float(fitted.aic)
                    best_order = order
            except Exception as exc:
                LOGGER.debug("ARMA fit failed for order=%s: %s", order, exc)

    if best_order is None:
        raise RuntimeError("No ARMA model fit succeeded for any tested (p, q) order.")

    return best_order, best_aic


def rolling_one_step_forecast(
    history: pd.Series,
    future: pd.Series,
    order: tuple[int, int, int],
) -> list[float]:
    """
    Perform walk-forward one-step-ahead forecasting with model re-fitting.

    Parameters:
        history: Historical series available at the start.
        future: Future series to forecast one step at a time.
        order: ARMA order as (p, d, q).

    Returns:
        List of forecasts on the transformed scale.
    """
    history_values = list(history.values.astype(float))
    predictions: list[float] = []

    for actual_value in future.values.astype(float):
        model = SARIMAX(
            endog=np.asarray(history_values, dtype=float),
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        forecast = float(fitted.forecast(steps=1)[0])

        predictions.append(forecast)
        history_values.append(float(actual_value))

    return predictions


def inverse_transform_predictions(
    predictions: list[float],
    use_log_transform: bool,
) -> np.ndarray:
    """
    Invert the transformation applied to predictions.

    Parameters:
        predictions: Predicted values on the transformed scale.
        use_log_transform: Whether log transformation had been applied.

    Returns:
        Predictions on the original scale as a NumPy array.
    """
    preds = np.asarray(predictions, dtype=float)
    if use_log_transform:
        return np.exp(preds)
    return preds


# --------------------------------------------------------------------------------------
# Main forecasting function
# --------------------------------------------------------------------------------------

def arma_forecast(
    dataset: pd.DataFrame,
    product: str,
    product_sign: str,
    price_type: str,
    output_folder: Path,
) -> ArmaRunResult:
    """
    Fit an ARMA model and generate validation/test forecasts for one configuration.

    Parameters:
        dataset: Full processed dataset.
        product: Time product, e.g. '00_04'.
        product_sign: Regulation direction, e.g. 'POS' or 'NEG'.
        price_type: Price type, e.g. 'AVERAGE' or 'MARGINAL'.
        output_folder: Directory where result files will be saved.

    Returns:
        ArmaRunResult with run metadata and status.
    """
    target_column = build_target_column(product_sign=product_sign, price_type=price_type)
    result_file = output_folder / f"arma_model_results_{product}_{product_sign}_{price_type}.csv"

    try:
        LOGGER.info(
            "Processing product=%s, sign=%s, price_type=%s",
            product,
            product_sign,
            price_type,
        )

        subset = dataset.loc[dataset["PRODUCT"] == product].copy()
        if subset.empty:
            raise ValueError(f"No data found for PRODUCT={product}")

        if target_column not in subset.columns:
            raise KeyError(f"Column not found: {target_column}")

        # Keep only target column and drop missing values safely
        target_df = subset[[target_column]].dropna().copy()
        target_df = ensure_daily_datetime_index(target_df)

        if len(target_df) < 10:
            raise ValueError(
                f"Not enough observations after filtering/dropping NaNs for {product}, "
                f"{product_sign}, {price_type}"
            )

        original_df = target_df.copy()

        # Split first, then fit preprocessing on train only
        train_raw, valid_raw, test_raw = split_time_series(target_df)
        train_t, valid_t, test_t, preprocessing_info = apply_preprocessing_from_train(
            train=train_raw,
            valid=valid_raw,
            test=test_raw,
            use_capping=True,
            use_log_transform=True,
        )

        # Select best order using training AIC
        best_order, best_aic = select_best_arma_order(train_series=train_t.iloc[:, 0])

        # Validation rolling forecast
        valid_predictions_t = rolling_one_step_forecast(
            history=train_t.iloc[:, 0],
            future=valid_t.iloc[:, 0],
            order=best_order,
        )

        # Test rolling forecast
        final_train_t = pd.concat([train_t, valid_t], axis=0)
        test_predictions_t = rolling_one_step_forecast(
            history=final_train_t.iloc[:, 0],
            future=test_t.iloc[:, 0],
            order=best_order,
        )

        # Inverse transform predictions to original scale
        valid_predictions = inverse_transform_predictions(
            predictions=valid_predictions_t,
            use_log_transform=bool(preprocessing_info["use_log_transform"]),
        )
        test_predictions = inverse_transform_predictions(
            predictions=test_predictions_t,
            use_log_transform=bool(preprocessing_info["use_log_transform"]),
        )

        # Build output DataFrame using original scale actuals
        valid_results_df = pd.DataFrame(
            {
                "DATE": valid_raw.index,
                "ACTUAL_VALUE": valid_raw.iloc[:, 0].values,
                "D+1": valid_predictions,
            }
        )

        test_results_df = pd.DataFrame(
            {
                "DATE": test_raw.index,
                "ACTUAL_VALUE": test_raw.iloc[:, 0].values,
                "D+1": test_predictions,
            }
        )

        results_df = pd.concat([valid_results_df, test_results_df], ignore_index=True)
        results_df.to_csv(result_file, index=False)

        LOGGER.info(
            "Completed product=%s, sign=%s, price_type=%s, order=%s, aic=%.4f",
            product,
            product_sign,
            price_type,
            best_order,
            best_aic,
        )

        return ArmaRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            order=str(best_order),
            aic=best_aic,
            output_file=str(result_file),
            success=True,
            message="Completed successfully",
        )

    except Exception as exc:
        LOGGER.exception(
            "Failed product=%s, sign=%s, price_type=%s: %s",
            product,
            product_sign,
            price_type,
            exc,
        )
        return ArmaRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            order="",
            aic=None,
            output_file=str(result_file),
            success=False,
            message=str(exc),
        )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main() -> None:
    """
    Main script entry point.

    Parameters:
        None

    Returns:
        None
    """
    configure_logging()

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Non-invertible starting MA parameters found.*",
    )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    products = ["00_04", "04_08", "08_12", "12_16", "16_20", "20_24"]
    product_signs = ["NEG", "POS"]
    price_types = ["AVERAGE", "MARGINAL"]

    output_folder = BASE_OUTPUT_PATH / "arma_model_output"
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_path = BASE_INPUT_PATH / "processed_afrr_data.csv"
    dataset = load_dataset(dataset_path=dataset_path)

    run_results = Parallel(n_jobs=-1)(
        delayed(arma_forecast)(
            dataset=dataset,
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            output_folder=output_folder,
        )
        for product in products
        for product_sign in product_signs
        for price_type in price_types
    )

    overview_rows = [
        {
            "PRODUCT": result.product,
            "PRODUCT_SIGN": result.product_sign,
            "PRICE_TYPE": result.price_type,
            "ORDER": result.order,
            "AIC": result.aic,
            "OUTPUT_FILE": result.output_file,
            "SUCCESS": result.success,
            "MESSAGE": result.message,
        }
        for result in run_results
    ]

    overview_df = pd.DataFrame(overview_rows).sort_values(
        by=["PRODUCT", "PRODUCT_SIGN", "PRICE_TYPE"]
    )
    overview_df.to_csv(output_folder / "arma_models_overview.csv", index=False)

    n_success = int(overview_df["SUCCESS"].sum()) if not overview_df.empty else 0
    LOGGER.info("Finished ARMA runs: %s/%s successful", n_success, len(overview_df))


# Example usage
# python3 -m statistical_methods.arma
if __name__ == "__main__":
    main()