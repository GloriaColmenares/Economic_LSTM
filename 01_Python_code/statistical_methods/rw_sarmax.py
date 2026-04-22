# 01_Python_code/statistical_methods/rw_sarmax.py

from __future__ import annotations

# Standard library imports
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------------------
# Limit native math-library threading to reduce worker crashes
# --------------------------------------------------------------------------------------

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Third-party imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pmdarima import auto_arima
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
class RwSarmaxRunResult:
    """
    Stores metadata for one rolling-window SARMAX run.

    Parameters:
        product: Time product identifier.
        product_sign: Regulation direction.
        price_type: Price type.
        window_months: Rolling window length in months.
        feature_numbers: Selected feature numbers.
        exogenous_factors: Exogenous feature names.
        num_features: Number of exogenous features.
        output_file: Path to the saved raw prediction CSV.
        success: Whether the run completed successfully.
        message: Informational or error message.
        n_predictions: Number of generated predictions.
    """

    product: str
    product_sign: str
    price_type: str
    window_months: int
    feature_numbers: str
    exogenous_factors: str
    num_features: int
    output_file: str
    success: bool
    message: str
    n_predictions: int


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
    return ensure_datetime_index(df)


def parse_feature_set_from_env(default: str = "4") -> list[int]:
    """
    Parse FEATURE_SET from environment.

    Expected examples:
        FEATURE_SET=1
        FEATURE_SET=4
        FEATURE_SET=1_4
        FEATURE_SET=1_3

    Parameters:
        default: Default feature set string if FEATURE_SET is not provided.

    Returns:
        Parsed list of feature integers.
    """
    feature_env = os.getenv("FEATURE_SET", default).strip()
    if not feature_env:
        feature_env = default

    try:
        feature_nums = [int(x) for x in feature_env.split("_")]
    except ValueError as exc:
        raise ValueError(
            f"Invalid FEATURE_SET='{feature_env}'. Expected format like '1_4' or '4'."
        ) from exc

    if not feature_nums:
        raise ValueError("Parsed FEATURE_SET is empty.")

    return feature_nums


def build_target_column(product_sign: str, price_type: str) -> str:
    """
    Build the target column name used in the dataset.

    Parameters:
        product_sign: Regulation direction.
        price_type: Price type.

    Returns:
        Target column name.
    """
    return f"{product_sign}_GERMANY_{price_type}_CAPACITY_PRICE_[(EUR/MW)/h]"


def preprocess_window(
    train_raw: pd.DataFrame,
    test_value_raw: float,
    use_capping: bool = True,
    use_log_transform: bool = True,
) -> tuple[pd.Series, float, float, bool]:
    """
    Preprocess one rolling train window and its next target value.

    Parameters:
        train_raw: Raw training window.
        test_value_raw: Raw next target value.
        use_capping: Whether to cap using train 95th percentile.
        use_log_transform: Whether to apply log transform.

    Returns:
        train_series_transformed, test_value_transformed, cap_value, used_log_transform
    """
    train_df = train_raw.copy()
    test_value = float(test_value_raw)
    cap_value = np.nan

    if use_capping:
        cap_value = float(train_df.quantile(0.95).iloc[0])
        train_df = train_df.clip(upper=cap_value, axis=1)
        test_value = min(test_value, cap_value)

    if use_log_transform:
        if (train_df.iloc[:, 0] <= 0).any():
            min_value = float(train_df.iloc[:, 0].min())
            raise ValueError(
                f"Rolling train window contains non-positive values; cannot apply log transform. "
                f"Minimum value: {min_value}"
            )
        if test_value <= 0:
            raise ValueError(
                f"Rolling test value is non-positive after capping; cannot apply log transform. "
                f"Value: {test_value}"
            )

        train_series = np.log(train_df.iloc[:, 0])
        test_value_t = float(np.log(test_value))
        return train_series, test_value_t, cap_value, True

    return train_df.iloc[:, 0], test_value, cap_value, False


def select_best_sarmax_model(
    train_series: pd.Series,
    exog_train: pd.DataFrame,
    seasonal_period: int = 7,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], float]:
    """
    Select best SARMAX specification using auto_arima on the target series.

    Parameters:
        train_series: Training target series.
        exog_train: Training exogenous features.
        seasonal_period: Seasonal period.

    Returns:
        best_order, best_seasonal_order, best_aic
    """
    try:
        best_model = auto_arima(
            y=train_series,
            d=0,
            D=0,
            m=seasonal_period,
            seasonal=True,
            stationary=True,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
    except Exception as exc:
        raise RuntimeError(f"auto_arima failed: {exc}") from exc

    best_order = tuple(best_model.order)
    best_seasonal_order = tuple(best_model.seasonal_order)
    best_aic = float(best_model.aic())

    return best_order, best_seasonal_order, best_aic


# --------------------------------------------------------------------------------------
# Main forecasting function
# --------------------------------------------------------------------------------------

def rw_sarmax_forecast(
    dataset: pd.DataFrame,
    exo_dataset: pd.DataFrame,
    product: str,
    product_sign: str,
    price_type: str,
    month: int,
    feature_nums: list[int],
    base_output_path: Path,
) -> RwSarmaxRunResult:
    """
    Fit a rolling-window SARMAX model and generate one-step-ahead forecasts.

    Parameters:
        dataset: Full processed target dataset.
        exo_dataset: Full processed exogenous dataset.
        product: Time product.
        product_sign: Regulation direction.
        price_type: Price type.
        month: Rolling window size in months.
        feature_nums: Selected exogenous feature numbers.
        base_output_path: Base output path.

    Returns:
        RwSarmaxRunResult
    """
    feature_nums_str = "_".join(map(str, feature_nums))
    output_folder = base_output_path / f"rw_sarmax_exog_{feature_nums_str}_{month}m_model_output"
    output_folder.mkdir(parents=True, exist_ok=True)

    result_file = output_folder / (
        f"rw_sarmax_exog_{feature_nums_str}_{month}m_model_results_{product}_{product_sign}_{price_type}.csv"
    )
    target_column = build_target_column(product_sign=product_sign, price_type=price_type)

    try:
        LOGGER.info(
            "Processing product=%s, sign=%s, price_type=%s, window=%sm, features=%s",
            product,
            product_sign,
            price_type,
            month,
            feature_nums_str,
        )

        subset_y = dataset.loc[dataset["PRODUCT"] == product].copy()
        if subset_y.empty:
            raise ValueError(f"No target data found for PRODUCT={product}")

        if target_column not in subset_y.columns:
            raise KeyError(f"Column not found: {target_column}")

        target_df = subset_y[[target_column]].dropna().copy()
        target_df = ensure_datetime_index(target_df)

        subset_x = exo_dataset.loc[exo_dataset["PRODUCT"] == product].copy()
        if subset_x.empty:
            raise ValueError(f"No exogenous data found for PRODUCT={product}")

        subset_x = subset_x.drop(columns=["PRODUCT"], errors="ignore")
        subset_x = ensure_datetime_index(subset_x)

        common_idx = target_df.index.intersection(subset_x.index)
        if common_idx.empty:
            raise ValueError(f"No overlapping timestamps for PRODUCT={product}")

        target_df = target_df.loc[common_idx].copy()
        subset_x = subset_x.loc[common_idx].copy()

        if subset_x.isna().any().any():
            raise ValueError("Exogenous features contain missing values after alignment.")

        if len(target_df) < 30:
            raise ValueError(
                f"Not enough observations after filtering/alignment for "
                f"{product}, {product_sign}, {price_type}"
            )

        train_limit = int(len(target_df) * 0.70)
        test_df = target_df.iloc[train_limit:].copy()

        if test_df.empty:
            raise ValueError("Test split is empty after 70% boundary.")

        exogenous_factors_str = ", ".join(subset_x.columns.tolist())
        num_features = len(subset_x.columns)

        result_rows: list[dict] = []

        for i in range(len(test_df)):
            test_start_date = test_df.index[i]

            train_raw = target_df.loc[
                test_start_date - pd.DateOffset(months=month):
                test_start_date - pd.DateOffset(days=1)
            ].copy()

            exo_train = subset_x.loc[
                test_start_date - pd.DateOffset(months=month):
                test_start_date - pd.DateOffset(days=1)
            ].copy()

            exo_forecast = subset_x.loc[[test_start_date]].copy() if test_start_date in subset_x.index else pd.DataFrame()

            if train_raw.empty or exo_train.empty:
                LOGGER.warning(
                    "Skipping %s because rolling train target/exog window is empty.",
                    test_start_date,
                )
                continue

            if len(train_raw) < 14 or len(exo_train) < 14:
                LOGGER.warning(
                    "Skipping %s because rolling train target/exog window is too short for seasonal modeling: y=%s, x=%s",
                    test_start_date,
                    len(train_raw),
                    len(exo_train),
                )
                continue

            if exo_forecast.empty:
                LOGGER.warning(
                    "Skipping %s because no exogenous data is available for forecast date.",
                    test_start_date,
                )
                continue

            train_common_idx = train_raw.index.intersection(exo_train.index)
            train_raw = train_raw.loc[train_common_idx].copy()
            exo_train = exo_train.loc[train_common_idx].copy()

            if len(train_raw) < 14:
                LOGGER.warning(
                    "Skipping %s because aligned rolling train window is too short after re-alignment: %s rows.",
                    test_start_date,
                    len(train_raw),
                )
                continue

            test_value_raw = float(target_df.loc[test_start_date].iloc[0])

            try:
                train_series_t, _, _, used_log_transform = preprocess_window(
                    train_raw=train_raw,
                    test_value_raw=test_value_raw,
                    use_capping=True,
                    use_log_transform=True,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Skipping %s due to preprocessing failure: %s",
                    test_start_date,
                    exc,
                )
                continue

            try:
                best_order, best_seasonal_order, best_aic = select_best_sarmax_model(
                    train_series=train_series_t,
                    exog_train=exo_train,
                    seasonal_period=7,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Skipping %s because SARMAX model selection failed: %s",
                    test_start_date,
                    exc,
                )
                continue

            try:
                final_model = SARIMAX(
                    endog=train_series_t,
                    exog=exo_train,
                    order=best_order,
                    seasonal_order=best_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                final_model_fit = final_model.fit(disp=False)
                yhat = float(final_model_fit.forecast(steps=1, exog=exo_forecast)[0])
            except Exception as exc:
                LOGGER.warning(
                    "Skipping %s because final SARMAX fit/forecast failed: %s",
                    test_start_date,
                    exc,
                )
                continue

            if used_log_transform:
                yhat_inv = float(np.exp(yhat))
            else:
                yhat_inv = yhat

            result_rows.append(
                {
                    "DATE": test_start_date,
                    "ACTUAL_VALUE": test_value_raw,
                    "D+1": yhat_inv,
                    "ORDER": str(best_order),
                    "SEASONAL_ORDER": str(best_seasonal_order),
                    "AIC": best_aic,
                    "EXOGENOUS_FACTORS": exogenous_factors_str,
                    "NUM_FEATURES": num_features,
                    "FEATURE_NUMBERS": feature_nums_str,
                }
            )

        results_df = pd.DataFrame(result_rows)
        results_df.to_csv(result_file, index=False)

        LOGGER.info(
            "Completed product=%s, sign=%s, price_type=%s, window=%sm, features=%s with %s predictions",
            product,
            product_sign,
            price_type,
            month,
            feature_nums_str,
            len(results_df),
        )

        return RwSarmaxRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            window_months=month,
            feature_numbers=feature_nums_str,
            exogenous_factors=exogenous_factors_str,
            num_features=num_features,
            output_file=str(result_file),
            success=True,
            message="Completed successfully",
            n_predictions=len(results_df),
        )

    except Exception as exc:
        LOGGER.exception(
            "Failed product=%s, sign=%s, price_type=%s, window=%sm, features=%s: %s",
            product,
            product_sign,
            price_type,
            month,
            feature_nums_str,
            exc,
        )
        return RwSarmaxRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            window_months=month,
            feature_numbers=feature_nums_str,
            exogenous_factors="",
            num_features=0,
            output_file=str(result_file),
            success=False,
            message=str(exc),
            n_predictions=0,
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
    months = [3, 6, 12]

    feature_nums = parse_feature_set_from_env(default="4")
    LOGGER.info("Using feature_nums=%s", feature_nums)

    exog_factors = {
        "feature_1": "FCR_GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]",
        "feature_2": "CUMULATED_CAPACITY_MORE_EXPENSIVE_THAN_ELECTRICITY_PRICE_EXCESS_CAPACITY",
        "feature_3": "RES_SHARE",
        "feature_4": "EXESSIVE_AVAILABLE_CAPACITY_UNTIL_PRICE_LIMIT_9999_€/MWh",
        "feature_5": "CLEAN_SPREAD_OF_GAS",
    }

    invalid_features = [n for n in feature_nums if f"feature_{n}" not in exog_factors]
    if invalid_features:
        raise KeyError(f"Invalid feature numbers requested: {invalid_features}")

    selected_exog_features = [exog_factors[f"feature_{num}"] for num in feature_nums]

    dataset_path = BASE_INPUT_PATH / "processed_afrr_data.csv"
    exo_dataset_path = BASE_INPUT_PATH / "selected_exogenous_factors_20210101_20250228.csv"

    processed_afrr_data = load_dataset(dataset_path)
    exogenous_factor_data = load_dataset(exo_dataset_path)

    required_exo_columns = ["PRODUCT"] + selected_exog_features
    missing_exo_columns = [col for col in required_exo_columns if col not in exogenous_factor_data.columns]
    if missing_exo_columns:
        raise KeyError(f"Missing exogenous columns: {missing_exo_columns}")

    exogenous_factor_data = exogenous_factor_data[required_exo_columns].copy()

    common_dates = processed_afrr_data.index.intersection(exogenous_factor_data.index)
    processed_afrr_data = processed_afrr_data.loc[common_dates].copy()
    exogenous_factor_data = exogenous_factor_data.loc[common_dates].copy()

    run_results = Parallel(n_jobs=2)(
        delayed(rw_sarmax_forecast)(
            dataset=processed_afrr_data,
            exo_dataset=exogenous_factor_data,
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            month=month,
            feature_nums=feature_nums,
            base_output_path=BASE_OUTPUT_PATH,
        )
        for product in products
        for product_sign in product_signs
        for price_type in price_types
        for month in months
    )

    overview_rows = [
        {
            "PRODUCT": result.product,
            "PRODUCT_SIGN": result.product_sign,
            "PRICE_TYPE": result.price_type,
            "WINDOW_MONTHS": result.window_months,
            "FEATURE_NUMBERS": result.feature_numbers,
            "EXOGENOUS_FACTORS": result.exogenous_factors,
            "NUM_FEATURES": result.num_features,
            "OUTPUT_FILE": result.output_file,
            "SUCCESS": result.success,
            "MESSAGE": result.message,
            "N_PREDICTIONS": result.n_predictions,
        }
        for result in run_results
    ]

    overview_df = pd.DataFrame(overview_rows).sort_values(
        by=["WINDOW_MONTHS", "PRODUCT", "PRODUCT_SIGN", "PRICE_TYPE"]
    )

    for month in months:
        feature_nums_str = "_".join(map(str, feature_nums))
        month_folder = BASE_OUTPUT_PATH / f"rw_sarmax_exog_{feature_nums_str}_{month}m_model_output"
        month_folder.mkdir(parents=True, exist_ok=True)

        month_overview = overview_df.loc[overview_df["WINDOW_MONTHS"] == month].copy()
        month_overview.to_csv(
            month_folder / f"rw_sarmax_exog_{feature_nums_str}_{month}m_models_overview.csv",
            index=False,
        )

        n_success = int(month_overview["SUCCESS"].sum()) if not month_overview.empty else 0
        LOGGER.info(
            "Finished rolling SARMAX runs for %sm: %s/%s successful",
            month,
            n_success,
            len(month_overview),
        )


# Example usage
# FEATURE_SET=1_3 python3 -m statistical_methods.rw_sarmax
if __name__ == "__main__":
    main()