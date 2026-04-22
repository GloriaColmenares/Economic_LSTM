# 01_Python_code/statistical_methods/armax.py

"""
ARMAX forecasting script for aFRR capacity prices.

This script:
- loads the processed target dataset
- loads selected exogenous variables
- fits ARMAX models for each product / sign / price type combination
- performs expanding-window one-step-ahead forecasting
- saves raw prediction results per combination
- saves one combined overview file after all runs finish

Notes:
- ARMAX is implemented through statsmodels' SARIMAX API with order=(p, 0, q)
- preprocessing parameters are fit on the training data only
- exogenous variables are aligned with the target series by timestamp and product
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
        level: Logging level.

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
class ArmaxRunResult:
    """
    Stores metadata for one ARMAX run.

    Parameters:
        product: Time product identifier.
        product_sign: Regulation direction.
        price_type: Price type.
        feature_numbers: Chosen exogenous feature numbers as a string.
        exogenous_factors: Names of the exogenous columns used.
        num_features: Number of exogenous variables.
        order: Best ARMAX order as a string.
        aic: Best AIC value.
        output_file: Path to the saved raw prediction CSV.
        success: Whether the run completed successfully.
        message: Informational or error message.
    """

    product: str
    product_sign: str
    price_type: str
    feature_numbers: str
    exogenous_factors: str
    num_features: int
    order: str
    aic: Optional[float]
    output_file: str
    success: bool
    message: str


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
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
    Build the target column name.

    Parameters:
        product_sign: Regulation direction.
        price_type: Price type.

    Returns:
        Target column name.
    """
    return f"{product_sign}_GERMANY_{price_type}_CAPACITY_PRICE_[(EUR/MW)/h]"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load a CSV dataset using DATE as DatetimeIndex.

    Parameters:
        dataset_path: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_csv(dataset_path, parse_dates=["DATE"], index_col="DATE")
    return ensure_datetime_index(df)


def split_time_series(
    y_df: pd.DataFrame,
    x_df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split aligned target and exogenous DataFrames into train/validation/test sets.

    Parameters:
        y_df: Target DataFrame.
        x_df: Exogenous DataFrame.
        train_ratio: Fraction used for training.
        valid_ratio: Fraction used for validation.

    Returns:
        Tuple:
        (y_train, y_valid, y_test, x_train, x_valid, x_test)

    Raises:
        ValueError: If the inputs are misaligned or any split is empty.
    """
    if len(y_df) != len(x_df):
        raise ValueError("Target and exogenous data must have the same number of rows.")

    if not y_df.index.equals(x_df.index):
        raise ValueError("Target and exogenous data must have identical indices.")

    n_obs = len(y_df)
    if n_obs < 10:
        raise ValueError("Not enough observations for train/validation/test split.")

    train_end = int(n_obs * train_ratio)
    valid_end = int(n_obs * (train_ratio + valid_ratio))

    y_train = y_df.iloc[:train_end].copy()
    y_valid = y_df.iloc[train_end:valid_end].copy()
    y_test = y_df.iloc[valid_end:].copy()

    x_train = x_df.iloc[:train_end].copy()
    x_valid = x_df.iloc[train_end:valid_end].copy()
    x_test = x_df.iloc[valid_end:].copy()

    if any(split.empty for split in [y_train, y_valid, y_test, x_train, x_valid, x_test]):
        raise ValueError(
            "Invalid split sizes for target/exogenous data. "
            f"y_train={len(y_train)}, y_valid={len(y_valid)}, y_test={len(y_test)}, "
            f"x_train={len(x_train)}, x_valid={len(x_valid)}, x_test={len(x_test)}"
        )

    return y_train, y_valid, y_test, x_train, x_valid, x_test


def apply_target_preprocessing_from_train(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    use_capping: bool = True,
    use_log_transform: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Apply target preprocessing using parameters estimated from the training set only.

    Parameters:
        train: Training target DataFrame.
        valid: Validation target DataFrame.
        test: Test target DataFrame.
        use_capping: Whether to cap values at the train 95th percentile.
        use_log_transform: Whether to apply a natural log transform.

    Returns:
        Tuple of transformed (train, valid, test, preprocessing_info).

    Raises:
        ValueError: If log transform is requested but non-positive values exist.
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


def validate_exogenous_data(x_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate exogenous variables.

    Parameters:
        x_df: Exogenous DataFrame.

    Returns:
        Cleaned exogenous DataFrame.

    Raises:
        ValueError: If exogenous data has missing or non-numeric values.
    """
    if x_df.empty:
        raise ValueError("Exogenous DataFrame is empty.")

    non_numeric_columns = x_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Exogenous columns must be numeric. Non-numeric columns: {non_numeric_columns}")

    if x_df.isna().any().any():
        missing_count = int(x_df.isna().sum().sum())
        raise ValueError(f"Exogenous data contains missing values: {missing_count}")

    return x_df.copy()


def select_best_armax_order(
    train_series: pd.Series,
    exog_train: pd.DataFrame,
    max_p: int = 4,
    max_q: int = 4,
) -> tuple[tuple[int, int, int], float]:
    """
    Select the best ARMAX order using training-set AIC.

    Parameters:
        train_series: Training target series.
        exog_train: Training exogenous variables.
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
                    exog=exog_train,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)
                if fitted.aic < best_aic:
                    best_aic = float(fitted.aic)
                    best_order = order
            except Exception as exc:
                LOGGER.debug("ARMAX fit failed for order=%s: %s", order, exc)

    if best_order is None:
        raise RuntimeError("No ARMAX model fit succeeded for any tested (p, q) order.")

    return best_order, best_aic


def rolling_one_step_forecast_armax(
    history_y: pd.Series,
    history_x: pd.DataFrame,
    future_y: pd.Series,
    future_x: pd.DataFrame,
    order: tuple[int, int, int],
) -> list[float]:
    """
    Perform expanding-window one-step-ahead forecasting for ARMAX.

    Parameters:
        history_y: Historical target series available at the start.
        history_x: Historical exogenous variables available at the start.
        future_y: Future target series, used to update the expanding window after each step.
        future_x: Future exogenous rows used for forecasting.
        order: ARMAX order as (p, d, q).

    Returns:
        List of forecasts on the transformed scale.

    Raises:
        ValueError: If target and exogenous inputs are misaligned.
    """
    if len(history_y) != len(history_x):
        raise ValueError("Initial history target and exogenous data must have the same length.")
    if len(future_y) != len(future_x):
        raise ValueError("Future target and exogenous data must have the same length.")

    history_y_values = list(history_y.values.astype(float))
    history_x_values = history_x.values.astype(float).tolist()
    predictions: list[float] = []

    for step_idx in range(len(future_y)):
        model = SARIMAX(
            endog=np.asarray(history_y_values, dtype=float),
            exog=np.asarray(history_x_values, dtype=float),
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)

        next_exog = future_x.iloc[[step_idx]].values.astype(float)
        forecast = float(fitted.forecast(steps=1, exog=next_exog)[0])
        predictions.append(forecast)

        history_y_values.append(float(future_y.iloc[step_idx]))
        history_x_values.append(future_x.iloc[step_idx].values.astype(float).tolist())

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
        Predictions on the original scale.
    """
    preds = np.asarray(predictions, dtype=float)
    if use_log_transform:
        return np.exp(preds)
    return preds


# --------------------------------------------------------------------------------------
# Main forecasting function
# --------------------------------------------------------------------------------------

def armax_forecast(
    dataset: pd.DataFrame,
    exo_dataset: pd.DataFrame,
    product: str,
    product_sign: str,
    price_type: str,
    feature_nums: list[int],
    output_folder: Path,
) -> ArmaxRunResult:
    """
    Fit an ARMAX model and generate validation/test forecasts for one configuration.

    Parameters:
        dataset: Full processed target dataset.
        exo_dataset: Full processed exogenous dataset.
        product: Time product.
        product_sign: Regulation direction.
        price_type: Price type.
        feature_nums: Selected feature numbers used in this run.
        output_folder: Directory where result files will be saved.

    Returns:
        ArmaxRunResult with run metadata and status.
    """
    target_column = build_target_column(product_sign=product_sign, price_type=price_type)
    feature_nums_str = "_".join(map(str, feature_nums))
    result_file = output_folder / (
        f"armax_exog_{feature_nums_str}_model_results_{product}_{product_sign}_{price_type}.csv"
    )

    try:
        LOGGER.info(
            "Processing product=%s, sign=%s, price_type=%s, features=%s",
            product,
            product_sign,
            price_type,
            feature_nums_str,
        )

        # Target subset
        subset_y = dataset.loc[dataset["PRODUCT"] == product].copy()
        if subset_y.empty:
            raise ValueError(f"No target data found for PRODUCT={product}")

        if target_column not in subset_y.columns:
            raise KeyError(f"Target column not found: {target_column}")

        subset_y = subset_y[[target_column]].copy()

        # Exogenous subset
        subset_x = exo_dataset.loc[exo_dataset["PRODUCT"] == product].copy()
        if subset_x.empty:
            raise ValueError(f"No exogenous data found for PRODUCT={product}")

        subset_x = subset_x.drop(columns=["PRODUCT"], errors="raise")
        subset_x = validate_exogenous_data(subset_x)

        # Align target and exogenous data on common timestamp index
        common_index = subset_y.index.intersection(subset_x.index)
        if common_index.empty:
            raise ValueError(f"No common timestamps between target and exogenous data for PRODUCT={product}")

        subset_y = subset_y.loc[common_index].sort_index()
        subset_x = subset_x.loc[common_index].sort_index()

        # Remove missing target rows and align exogenous rows accordingly
        valid_mask = subset_y[target_column].notna()
        subset_y = subset_y.loc[valid_mask].copy()
        subset_x = subset_x.loc[valid_mask].copy()

        subset_y = ensure_datetime_index(subset_y)
        subset_x = ensure_datetime_index(subset_x)

        if len(subset_y) < 10:
            raise ValueError(
                f"Not enough observations after filtering/alignment for "
                f"{product}, {product_sign}, {price_type}"
            )

        original_y = subset_y.copy()
        exogenous_factors_str = ", ".join(subset_x.columns.tolist())

        # Split first
        y_train_raw, y_valid_raw, y_test_raw, x_train, x_valid, x_test = split_time_series(
            y_df=subset_y,
            x_df=subset_x,
        )

        # Preprocess target only
        y_train_t, y_valid_t, y_test_t, preprocessing_info = apply_target_preprocessing_from_train(
            train=y_train_raw,
            valid=y_valid_raw,
            test=y_test_raw,
            use_capping=True,
            use_log_transform=True,
        )

        # Select best order
        best_order, best_aic = select_best_armax_order(
            train_series=y_train_t.iloc[:, 0],
            exog_train=x_train,
        )

        # Validation forecasts
        valid_predictions_t = rolling_one_step_forecast_armax(
            history_y=y_train_t.iloc[:, 0],
            history_x=x_train,
            future_y=y_valid_t.iloc[:, 0],
            future_x=x_valid,
            order=best_order,
        )

        # Test forecasts
        y_final_train_t = pd.concat([y_train_t, y_valid_t], axis=0)
        x_final_train = pd.concat([x_train, x_valid], axis=0)

        test_predictions_t = rolling_one_step_forecast_armax(
            history_y=y_final_train_t.iloc[:, 0],
            history_x=x_final_train,
            future_y=y_test_t.iloc[:, 0],
            future_x=x_test,
            order=best_order,
        )

        # Inverse transform predictions
        valid_predictions = inverse_transform_predictions(
            predictions=valid_predictions_t,
            use_log_transform=bool(preprocessing_info["use_log_transform"]),
        )
        test_predictions = inverse_transform_predictions(
            predictions=test_predictions_t,
            use_log_transform=bool(preprocessing_info["use_log_transform"]),
        )

        # Build output DataFrame
        valid_results_df = pd.DataFrame(
            {
                "DATE": y_valid_raw.index,
                "ACTUAL_VALUE": y_valid_raw.iloc[:, 0].values,
                "D+1": valid_predictions,
            }
        )

        test_results_df = pd.DataFrame(
            {
                "DATE": y_test_raw.index,
                "ACTUAL_VALUE": y_test_raw.iloc[:, 0].values,
                "D+1": test_predictions,
            }
        )

        results_df = pd.concat([valid_results_df, test_results_df], ignore_index=True)
        results_df.to_csv(result_file, index=False)

        LOGGER.info(
            "Completed product=%s, sign=%s, price_type=%s, features=%s, order=%s, aic=%.4f",
            product,
            product_sign,
            price_type,
            feature_nums_str,
            best_order,
            best_aic,
        )

        return ArmaxRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            feature_numbers=feature_nums_str,
            exogenous_factors=exogenous_factors_str,
            num_features=len(subset_x.columns),
            order=str(best_order),
            aic=best_aic,
            output_file=str(result_file),
            success=True,
            message="Completed successfully",
        )

    except Exception as exc:
        LOGGER.exception(
            "Failed product=%s, sign=%s, price_type=%s, features=%s: %s",
            product,
            product_sign,
            price_type,
            feature_nums_str,
            exc,
        )
        return ArmaxRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            feature_numbers=feature_nums_str,
            exogenous_factors="",
            num_features=0,
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

    #feature_nums = [4]
    feature_nums = [1]
    #feature_nums = [1,4]
    #feature_nums = [1,3]


    exog_factors = {
        "feature_1": "FCR_GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]",
        "feature_2": "CUMULATED_CAPACITY_MORE_EXPENSIVE_THAN_ELECTRICITY_PRICE_EXCESS_CAPACITY",
        "feature_3": "RES_SHARE",
        "feature_4": "EXESSIVE_AVAILABLE_CAPACITY_UNTIL_PRICE_LIMIT_9999_€/MWh",
        "feature_5": "CLEAN_SPREAD_OF_GAS",
    }

    selected_exog_features = [exog_factors[f"feature_{num}"] for num in feature_nums]
    feature_nums_str = "_".join(map(str, feature_nums))

    output_folder = BASE_OUTPUT_PATH / f"armax_exog_{feature_nums_str}_model_output"
    output_folder.mkdir(parents=True, exist_ok=True)

    target_dataset_path = BASE_INPUT_PATH / "processed_afrr_data.csv"
    exo_dataset_path = BASE_INPUT_PATH / "selected_exogenous_factors_20210101_20250228.csv"

    dataset = load_dataset(dataset_path=target_dataset_path)
    exo_dataset = load_dataset(dataset_path=exo_dataset_path)

    required_exo_columns = ["PRODUCT"] + selected_exog_features
    missing_exo_columns = [col for col in required_exo_columns if col not in exo_dataset.columns]
    if missing_exo_columns:
        raise KeyError(f"Missing exogenous columns: {missing_exo_columns}")

    exo_dataset = exo_dataset[required_exo_columns].copy()

    run_results = Parallel(n_jobs=-1)(
        delayed(armax_forecast)(
            dataset=dataset,
            exo_dataset=exo_dataset,
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            feature_nums=feature_nums,
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
            "EXOGENOUS_FACTORS": result.exogenous_factors,
            "NUM_FEATURES": result.num_features,
            "FEATURE_NUMBERS": result.feature_numbers,
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
    overview_df.to_csv(
        output_folder / f"armax_exog_{feature_nums_str}_models_overview.csv",
        index=False,
    )

    n_success = int(overview_df["SUCCESS"].sum()) if not overview_df.empty else 0
    LOGGER.info("Finished ARMAX runs: %s/%s successful", n_success, len(overview_df))


# Example usage
# python3 -m statistical_methods.armax
if __name__ == "__main__":
    main()