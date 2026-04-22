# 01_Python_code/statistical_methods/rw_arma.py

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
class RwArmaRunResult:
    """
    Stores metadata for one rolling-window ARMA run.

    Parameters:
        product: Time product identifier.
        product_sign: Regulation direction.
        price_type: Price type.
        window_months: Rolling window length in months.
        output_file: Path to the saved raw prediction CSV.
        success: Whether the run completed successfully.
        message: Informational or error message.
        n_predictions: Number of generated predictions.
    """

    product: str
    product_sign: str
    price_type: str
    window_months: int
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


def select_best_arma_order(
    train_series: pd.Series,
    max_p: int = 3,
    max_q: int = 3,
) -> tuple[tuple[int, int, int], float]:
    """
    Select the best ARMA order using AIC.

    Parameters:
        train_series: Training target series.
        max_p: Maximum AR order.
        max_q: Maximum MA order.

    Returns:
        best_order, best_aic

    Raises:
        RuntimeError: If all model fits fail.
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
        raise RuntimeError("No ARMA model fit succeeded for the current rolling window.")

    return best_order, best_aic


def preprocess_window(
    train_raw: pd.DataFrame,
    test_value_raw: float,
    use_capping: bool = True,
    use_log_transform: bool = True,
) -> tuple[pd.Series, float, float, bool]:
    """
    Preprocess one rolling train window and its corresponding next test value.

    Parameters:
        train_raw: Raw training window.
        test_value_raw: Raw next target value.
        use_capping: Whether to cap using the train 95th percentile.
        use_log_transform: Whether to apply log transform.

    Returns:
        train_series_transformed, test_value_transformed, cap_value, used_log_transform

    Raises:
        ValueError: If non-positive values prevent log transformation.
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


# --------------------------------------------------------------------------------------
# Main forecasting function
# --------------------------------------------------------------------------------------

def rw_arma_forecast(
    dataset: pd.DataFrame,
    product: str,
    product_sign: str,
    price_type: str,
    month: int,
    base_output_path: Path,
) -> RwArmaRunResult:
    """
    Fit a rolling-window ARMA model and generate one-step-ahead forecasts.

    Parameters:
        dataset: Full processed dataset.
        product: Time product.
        product_sign: Regulation direction.
        price_type: Price type.
        month: Rolling window size in months.
        base_output_path: Base output path.

    Returns:
        RwArmaRunResult
    """
    output_folder = base_output_path / f"rw_arma_{month}m_model_output"
    output_folder.mkdir(parents=True, exist_ok=True)

    result_file = output_folder / f"rw_arma_{month}m_model_results_{product}_{product_sign}_{price_type}.csv"
    target_column = build_target_column(product_sign=product_sign, price_type=price_type)

    try:
        LOGGER.info(
            "Processing product=%s, sign=%s, price_type=%s, window=%sm",
            product,
            product_sign,
            price_type,
            month,
        )

        subset = dataset.loc[dataset["PRODUCT"] == product].copy()
        if subset.empty:
            raise ValueError(f"No data found for PRODUCT={product}")

        if target_column not in subset.columns:
            raise KeyError(f"Column not found: {target_column}")

        # Safe target extraction
        df = subset[[target_column]].dropna().copy()
        df = ensure_datetime_index(df)

        if len(df) < 20:
            raise ValueError(
                f"Not enough observations after filtering for {product}, {product_sign}, {price_type}"
            )

        # Rolling test starts after initial 70%
        train_limit = int(len(df) * 0.70)
        test_df = df.iloc[train_limit:].copy()

        if test_df.empty:
            raise ValueError("Test split is empty after 70% boundary.")

        result_rows: list[dict] = []

        for i in range(len(test_df)):
            test_start_date = test_df.index[i]

            # Fixed-size rolling window
            train_raw = df.loc[
                test_start_date - pd.DateOffset(months=month):
                test_start_date - pd.DateOffset(days=1)
            ].copy()

            if train_raw.empty:
                LOGGER.warning(
                    "Skipping %s because rolling train window is empty.",
                    test_start_date,
                )
                continue

            if len(train_raw) < 10:
                LOGGER.warning(
                    "Skipping %s because rolling train window is too short: %s rows.",
                    test_start_date,
                    len(train_raw),
                )
                continue

            test_value_raw = float(df.loc[test_start_date].iloc[0])

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
                best_order, best_aic = select_best_arma_order(
                    train_series=train_series_t,
                    max_p=3,
                    max_q=3,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Skipping %s because no ARMA order fit succeeded: %s",
                    test_start_date,
                    exc,
                )
                continue

            try:
                final_model = SARIMAX(
                    endog=train_series_t,
                    order=best_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                final_model_fit = final_model.fit(disp=False)
                yhat = float(final_model_fit.forecast(steps=1)[0])
            except Exception as exc:
                LOGGER.warning(
                    "Skipping %s because final ARMA fit/forecast failed: %s",
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
                    "AIC": best_aic,
                }
            )

        results_df = pd.DataFrame(result_rows)
        results_df.to_csv(result_file, index=False)

        LOGGER.info(
            "Completed product=%s, sign=%s, price_type=%s, window=%sm with %s predictions",
            product,
            product_sign,
            price_type,
            month,
            len(results_df),
        )

        return RwArmaRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            window_months=month,
            output_file=str(result_file),
            success=True,
            message="Completed successfully",
            n_predictions=len(results_df),
        )

    except Exception as exc:
        LOGGER.exception(
            "Failed product=%s, sign=%s, price_type=%s, window=%sm: %s",
            product,
            product_sign,
            price_type,
            month,
            exc,
        )
        return RwArmaRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            window_months=month,
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

    dataset_path = BASE_INPUT_PATH / "processed_afrr_data.csv"
    processed_afrr_data = load_dataset(dataset_path)

    run_results = Parallel(n_jobs=2)(
        delayed(rw_arma_forecast)(
            dataset=processed_afrr_data,
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            month=month,
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
        month_folder = BASE_OUTPUT_PATH / f"rw_arma_{month}m_model_output"
        month_folder.mkdir(parents=True, exist_ok=True)

        month_overview = overview_df.loc[overview_df["WINDOW_MONTHS"] == month].copy()
        month_overview.to_csv(month_folder / f"rw_arma_{month}m_models_overview.csv", index=False)

        n_success = int(month_overview["SUCCESS"].sum()) if not month_overview.empty else 0
        LOGGER.info(
            "Finished rolling ARMA runs for %sm: %s/%s successful",
            month,
            n_success,
            len(month_overview),
        )


# Example usage
# python3 -m statistical_methods.rw_arma
if __name__ == "__main__":
    main()