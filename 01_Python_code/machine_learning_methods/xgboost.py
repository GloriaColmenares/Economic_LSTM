# 01_Python_code/machine_learning_methods/xgboost_features_v1.py

from __future__ import annotations

# Standard library imports
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBRegressor

# --------------------------------------------------------------------------------------
# Import project paths safely
# --------------------------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
PROJECT_CODE_ROOT = CURRENT_FILE.parents[1]  # 01_Python_code
if str(PROJECT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_ROOT))

from paths import BASE_INPUT_PATH, BASE_OUTPUT_PATH  # noqa: E402

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging once.

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
# Data class
# --------------------------------------------------------------------------------------

@dataclass
class XGBoostRunResult:
    """
    Stores metadata for one XGBoost run.

    Parameters:
        product: Time product identifier.
        product_sign: Regulation direction.
        price_type: Price type.
        feature_numbers: Exogenous feature numbers.
        exogenous_factors: Exogenous feature names.
        num_features: Number of exogenous features.
        n_estimators: Number of trees.
        learning_rate: Learning rate.
        max_depth: Tree depth.
        output_file: Path to saved raw result file.
        success: Whether the run completed successfully.
        message: Informational or error message.
    """

    product: str
    product_sign: str
    price_type: str
    feature_numbers: str
    exogenous_factors: str
    num_features: int
    n_estimators: Optional[int]
    learning_rate: Optional[float]
    max_depth: Optional[int]
    output_file: str
    success: bool
    message: str


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and sort DatetimeIndex.

    Parameters:
        df: Input DataFrame.

    Returns:
        Sorted DataFrame.

    Raises:
        TypeError: If index is not DatetimeIndex.
        ValueError: If duplicates exist.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in index.")
    return df.sort_index()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load CSV dataset with DATE index.

    Parameters:
        dataset_path: Path to CSV.

    Returns:
        Loaded DataFrame.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    df = pd.read_csv(dataset_path, parse_dates=["DATE"], index_col="DATE")
    return ensure_datetime_index(df)


def parse_feature_set_from_env(default: str = "1_4") -> list[int]:
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
    Build target column name.

    Parameters:
        product_sign: Regulation direction.
        price_type: Price type.

    Returns:
        Target column name.
    """
    return f"{product_sign}_GERMANY_{price_type}_CAPACITY_PRICE_[(EUR/MW)/h]"


def make_xgb_safe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names for XGBoost compatibility.

    Parameters:
        df: Input DataFrame.

    Returns:
        DataFrame with safe column names.
    """
    df = df.copy()
    seen: dict[str, int] = {}
    safe_cols: list[str] = []

    for col in df.columns:
        c = str(col)
        c = re.sub(r"[\[\]<>]", "_", c)
        c = re.sub(r"[^0-9A-Za-z_]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        if not c:
            c = "feature"

        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0

        safe_cols.append(c)

    df.columns = safe_cols
    return df


def split_time_series_indices(n_obs: int) -> tuple[int, int]:
    """
    Compute train and validation boundaries.

    Parameters:
        n_obs: Number of observations.

    Returns:
        train_end, valid_end
    """
    if n_obs < 20:
        raise ValueError(f"Not enough observations for XGBoost split: {n_obs}")
    train_end = int(n_obs * 0.70)
    valid_end = int(n_obs * 0.85)
    return train_end, valid_end


def build_features(
    df_target: pd.DataFrame,
    df_exog: Optional[pd.DataFrame] = None,
    lags: tuple[int, ...] = (1, 2, 3, 7, 14, 28),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build supervised learning table for one-step-ahead forecasting.

    Parameters:
        df_target: Transformed target DataFrame with one column.
        df_exog: Optional exogenous feature DataFrame.
        lags: Lag values to create.

    Returns:
        X, y
    """
    feat = pd.DataFrame(index=df_target.index)
    feat["y"] = df_target.iloc[:, 0]

    for lag in lags:
        feat[f"lag_{lag}"] = feat["y"].shift(lag)

    feat["dow"] = feat.index.dayofweek
    feat["month"] = feat.index.month

    if df_exog is not None and not df_exog.empty:
        for col in df_exog.columns:
            feat[col] = df_exog[col]

    feat = feat.dropna()

    X = feat.drop(columns=["y"])
    y = feat["y"]
    return X, y


def apply_target_preprocessing_from_train(
    target_df: pd.DataFrame,
    train_end: int,
    valid_end: int,
    use_capping: bool = True,
    use_log_transform: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply target preprocessing with train-fitted parameters only.

    Parameters:
        target_df: Full aligned target DataFrame.
        train_end: Train boundary.
        valid_end: Validation boundary.
        use_capping: Whether to cap target using train 95th percentile.
        use_log_transform: Whether to log-transform target.

    Returns:
        transformed target DataFrame, preprocessing info
    """
    target_t = target_df.copy()
    preprocessing_info = {
        "use_capping": use_capping,
        "use_log_transform": use_log_transform,
        "cap_value": None,
    }

    train_df = target_t.iloc[:train_end].copy()
    valid_df = target_t.iloc[train_end:valid_end].copy()
    test_df = target_t.iloc[valid_end:].copy()

    if use_capping:
        cap_value = float(train_df.quantile(0.95).iloc[0])
        preprocessing_info["cap_value"] = cap_value

        train_df = train_df.clip(upper=cap_value, axis=1)
        valid_df = valid_df.clip(upper=cap_value, axis=1)
        test_df = test_df.clip(upper=cap_value, axis=1)

    if use_log_transform:
        for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            if (split_df.iloc[:, 0] <= 0).any():
                min_value = float(split_df.iloc[:, 0].min())
                raise ValueError(
                    f"Log transform requested but {split_name} contains non-positive values. "
                    f"Minimum value found: {min_value}"
                )

        train_df = np.log(train_df)
        valid_df = np.log(valid_df)
        test_df = np.log(test_df)

    transformed = pd.concat([train_df, valid_df, test_df], axis=0)
    return transformed, preprocessing_info


# --------------------------------------------------------------------------------------
# Main model function
# --------------------------------------------------------------------------------------

def xgboost_forecast(
    dataset: pd.DataFrame,
    exo_dataset: Optional[pd.DataFrame],
    product: str,
    product_sign: str,
    price_type: str,
    feature_nums: list[int],
    output_folder: Path,
) -> XGBoostRunResult:
    """
    Fit XGBoost and generate validation/test forecasts for one configuration.

    Parameters:
        dataset: Main processed dataset.
        exo_dataset: Optional exogenous dataset.
        product: Time product.
        product_sign: Regulation direction.
        price_type: Price type.
        feature_nums: Selected exogenous feature numbers.
        output_folder: Output folder.

    Returns:
        XGBoostRunResult
    """
    feature_nums_str = "_".join(map(str, feature_nums))
    target_column = build_target_column(product_sign=product_sign, price_type=price_type)
    result_file = output_folder / (
        f"xgboost_exog_{feature_nums_str}_model_results_{product}_{product_sign}_{price_type}.csv"
    )

    try:
        LOGGER.info(
            "Processing product=%s, sign=%s, price_type=%s, features=%s",
            product,
            product_sign,
            price_type,
            feature_nums_str,
        )

        subset = dataset.loc[dataset["PRODUCT"] == product].copy()
        if subset.empty:
            raise ValueError(f"No data found for PRODUCT={product}")

        if target_column not in subset.columns:
            raise KeyError(f"Column not found: {target_column}")

        target_df = subset[[target_column]].dropna().copy()
        target_df = ensure_datetime_index(target_df)

        exo_product: Optional[pd.DataFrame] = None
        exogenous_factors_str = ""
        num_features = 0

        if exo_dataset is not None:
            exo_product = exo_dataset.loc[exo_dataset["PRODUCT"] == product].copy()
            if exo_product.empty:
                raise ValueError(f"No exogenous data found for PRODUCT={product}")

            exo_product = exo_product.drop(columns=["PRODUCT"], errors="ignore")
            exo_product = ensure_datetime_index(exo_product)

            common_idx = target_df.index.intersection(exo_product.index)
            if common_idx.empty:
                raise ValueError(f"No overlapping timestamps for PRODUCT={product}")

            target_df = target_df.loc[common_idx].copy()
            exo_product = exo_product.loc[common_idx].copy()

            if exo_product.isna().any().any():
                raise ValueError("Exogenous features contain missing values after alignment.")

            exogenous_factors_str = ", ".join(exo_product.columns.tolist())
            num_features = len(exo_product.columns)

        if len(target_df) < 20:
            raise ValueError(
                f"Not enough observations after filtering/alignment for "
                f"{product}, {product_sign}, {price_type}"
            )

        # Split boundaries on aligned raw target
        train_end_raw, valid_end_raw = split_time_series_indices(len(target_df))

        # Preprocess target using train only
        target_transformed, preprocessing_info = apply_target_preprocessing_from_train(
            target_df=target_df,
            train_end=train_end_raw,
            valid_end=valid_end_raw,
            use_capping=True,
            use_log_transform=True,
        )

        # Build supervised table after transformation
        X, y = build_features(
            df_target=target_transformed,
            df_exog=exo_product,
            lags=(1, 2, 3, 7, 14, 28),
        )

        X = make_xgb_safe_columns(X)

        n = len(X)
        train_end, valid_end = split_time_series_indices(n)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_valid, y_valid = X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]
        X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:]

        if X_train.empty or X_valid.empty or X_test.empty:
            raise ValueError(
                f"Invalid feature split sizes: train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}"
            )

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            early_stopping_rounds=50,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)

        if preprocessing_info["use_log_transform"]:
            valid_pred_inv = np.exp(valid_pred)
            test_pred_inv = np.exp(test_pred)
        else:
            valid_pred_inv = valid_pred
            test_pred_inv = test_pred

        valid_dates = X_valid.index
        test_dates = X_test.index

        valid_actual = target_df.loc[valid_dates].iloc[:, 0].values
        test_actual = target_df.loc[test_dates].iloc[:, 0].values

        valid_results_df = pd.DataFrame(
            {
                "DATE": valid_dates,
                "ACTUAL_VALUE": valid_actual,
                "D+1": valid_pred_inv,
            }
        )

        test_results_df = pd.DataFrame(
            {
                "DATE": test_dates,
                "ACTUAL_VALUE": test_actual,
                "D+1": test_pred_inv,
            }
        )

        results_df = pd.concat([valid_results_df, test_results_df], ignore_index=True)
        results_df.to_csv(result_file, index=False)

        LOGGER.info(
            "Completed product=%s, sign=%s, price_type=%s, features=%s",
            product,
            product_sign,
            price_type,
            feature_nums_str,
        )

        return XGBoostRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            feature_numbers=feature_nums_str,
            exogenous_factors=exogenous_factors_str,
            num_features=num_features,
            n_estimators=int(model.get_params()["n_estimators"]),
            learning_rate=float(model.get_params()["learning_rate"]),
            max_depth=int(model.get_params()["max_depth"]),
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
        return XGBoostRunResult(
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            feature_numbers=feature_nums_str,
            exogenous_factors="",
            num_features=0,
            n_estimators=None,
            learning_rate=None,
            max_depth=None,
            output_file=str(result_file),
            success=False,
            message=str(exc),
        )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point.

    Parameters:
        None

    Returns:
        None
    """
    configure_logging()

    products = ["00_04", "04_08", "08_12", "12_16", "16_20", "20_24"]
    product_signs = ["NEG", "POS"]
    price_types = ["AVERAGE", "MARGINAL"]

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

    selected_exog_features = [exog_factors[f"feature_{n}"] for n in feature_nums]
    feature_nums_str = "_".join(map(str, feature_nums))

    dataset_path = BASE_INPUT_PATH / "processed_afrr_data.csv"
    exo_dataset_path = BASE_INPUT_PATH / "selected_exogenous_factors_20210101_20250228.csv"

    dataset = load_dataset(dataset_path)
    exogenous_factor_data = load_dataset(exo_dataset_path)

    required_exo_columns = ["PRODUCT"] + selected_exog_features
    missing_exo_columns = [col for col in required_exo_columns if col not in exogenous_factor_data.columns]
    if missing_exo_columns:
        raise KeyError(f"Missing exogenous columns: {missing_exo_columns}")

    exogenous_factor_data = exogenous_factor_data[required_exo_columns].copy()

    common_dates = dataset.index.intersection(exogenous_factor_data.index)
    dataset = dataset.loc[common_dates].copy()
    exogenous_factor_data = exogenous_factor_data.loc[common_dates].copy()

    output_folder = BASE_OUTPUT_PATH / f"xgboost_exog_{feature_nums_str}_model_output"
    output_folder.mkdir(parents=True, exist_ok=True)

    run_results = Parallel(n_jobs=1)(
        delayed(xgboost_forecast)(
            dataset=dataset,
            exo_dataset=exogenous_factor_data,
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
            "N_ESTIMATORS": result.n_estimators,
            "LEARNING_RATE": result.learning_rate,
            "MAX_DEPTH": result.max_depth,
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
        output_folder / f"xgboost_exog_{feature_nums_str}_models_overview.csv",
        index=False,
    )

    n_success = int(overview_df["SUCCESS"].sum()) if not overview_df.empty else 0
    LOGGER.info("Finished XGBoost runs: %s/%s successful", n_success, len(overview_df))


# Example usage
# FEATURE_SET=1_4 python3 -m machine_learning_methods.xgboost_features_v1
if __name__ == "__main__":
    main()