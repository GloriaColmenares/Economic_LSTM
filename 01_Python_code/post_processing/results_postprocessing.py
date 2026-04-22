# 01_Python_code/postprocessing/results_postprocessing.py

from __future__ import annotations

# Standard library imports
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
CURRENT_FILE = Path(__file__).resolve()
PROJECT_CODE_ROOT = CURRENT_FILE.parents[1]  # 01_Python_code
if str(PROJECT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_ROOT))

from paths import BASE_INPUT_PATH, BASE_OUTPUT_PATH  # noqa: E402


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
# Data classes
# --------------------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """
    Stores metadata needed to locate and process one model family.

    Parameters:
        model_key: Canonical internal model key.
        folder_name: Folder inside BASE_OUTPUT_PATH.
        file_prefix: Prefix used in result CSV filenames.
        group: High-level model group.
        lookbacks: Optional list of lookbacks.
        strategies: Optional list of LSTM-style prediction strategies.
    """

    model_key: str
    folder_name: str
    file_prefix: str
    group: str
    lookbacks: Optional[list[int]] = None
    strategies: Optional[list[str]] = None


# --------------------------------------------------------------------------------------
# Helper functions
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


def load_results_file(results_file: Path) -> pd.DataFrame:
    """
    Load one results file using the strict DATE standard.

    Parameters:
        results_file: Path to result CSV.

    Returns:
        Loaded DataFrame indexed by DATE.

    Raises:
        KeyError: If DATE column is missing.
    """
    df = pd.read_csv(results_file)

    if "DATE" not in df.columns:
        raise KeyError(f"Expected 'DATE' column not found in {results_file}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.loc[df["DATE"].notna()].copy()
    df = df.set_index("DATE")
    return ensure_datetime_index(df)


def robust_normalize(series: pd.Series, upper_percentile: float = 99.0) -> pd.Series:
    """
    Normalize values where lower is better, with percentile capping.

    Parameters:
        series: Metric series.
        upper_percentile: Upper percentile used for capping.

    Returns:
        Normalized score where higher is better.
    """
    clean = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if clean.dropna().empty:
        return pd.Series(np.nan, index=series.index)

    cap_value = clean.quantile(upper_percentile / 100.0)
    if pd.isna(cap_value) or cap_value == 0:
        return pd.Series(np.nan, index=series.index)

    capped_series = clean.clip(upper=cap_value)
    return 1.0 - (capped_series / cap_value)


def build_marginal_column(product_sign: str) -> str:
    """
    Build marginal price column name.

    Parameters:
        product_sign: NEG or POS.

    Returns:
        Marginal price column name.
    """
    return f"{product_sign}_GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]"


def get_result_file_path(
    base_output_path: Path,
    spec: ModelSpec,
    product: str,
    product_sign: str,
    price_type: str,
    lookback: Optional[int] = None,
) -> Path:
    """
    Build the expected result file path for one configuration.

    Important:
        XGBoost v1 and XGBoost v2 use different filename prefixes in your outputs.
        This function handles both correctly.

    Parameters:
        base_output_path: Base output folder.
        spec: Model specification.
        product: Product block.
        product_sign: NEG or POS.
        price_type: AVERAGE or MARGINAL.
        lookback: Optional lookback.

    Returns:
        Expected result file path.
    """
    model_folder = base_output_path / spec.folder_name

    # XGBoost v2 keeps the full xgboost_v2_* prefix in the filename
    if spec.model_key.startswith("xgboost_v2_exog_"):
        if lookback is None:
            raise ValueError(f"lookback must be provided for model {spec.model_key}")

        filename = (
            f"{spec.file_prefix}_ts{lookback}_model_results_"
            f"{product}_{product_sign}_{price_type}.csv"
        )
        return model_folder / filename

    # XGBoost v1 has no lookback in the filename
    if spec.model_key.startswith("xgboost_exog_"):
        filename = (
            f"{spec.file_prefix}_model_results_"
            f"{product}_{product_sign}_{price_type}.csv"
        )
        return model_folder / filename

    # LSTM-style models include lookback in the filename
    if lookback is not None:
        filename = (
            f"{spec.file_prefix}_ts{lookback}_model_results_"
            f"{product}_{product_sign}_{price_type}.csv"
        )
    else:
        filename = (
            f"{spec.file_prefix}_model_results_"
            f"{product}_{product_sign}_{price_type}.csv"
        )

    return model_folder / filename


def select_prediction_column(
    results: pd.DataFrame,
    model_key: str,
    strategy: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardize the selected prediction column to D+1.

    Parameters:
        results: Results DataFrame.
        model_key: Internal model key.
        strategy: LSTM-like strategy if relevant.

    Returns:
        DataFrame with D+1 column.

    Raises:
        ValueError: If no usable prediction column is found.
    """
    results = results.copy()

    lstm_like_models = {
        "lstm",
        "bidirectional_lstm",
        "lstm_features_1",
        "lstm_features_4",
        "lstm_features_1_3",
        "lstm_features_1_4",
    }

    if model_key in lstm_like_models:
        strategy_to_column = {
            "ensemble": "D+1_ENSEMBLE_MODEL",
            "recency": "D+1_RECENCY_WEIGHTED",
            "single": "D+1_SINGLE_MODEL",
        }

        selected_col = strategy_to_column.get(strategy)
        if selected_col is None:
            raise ValueError(f"Invalid LSTM strategy '{strategy}' for model {model_key}")
        if selected_col not in results.columns:
            raise ValueError(
                f"Expected prediction column '{selected_col}' not found for model {model_key}"
            )

        results["D+1"] = results[selected_col]
        return results

    if "D+1" not in results.columns:
        raise ValueError(f"Expected 'D+1' column not found for model {model_key}")

    return results


def compute_processed_results(
    dataset: pd.DataFrame,
    results: pd.DataFrame,
    product: str,
    product_sign: str,
) -> pd.DataFrame:
    """
    Add metrics and commercial columns to one result DataFrame.

    Parameters:
        dataset: Full processed target dataset.
        results: Standardized results DataFrame with DATE index and D+1.
        product: Product block.
        product_sign: NEG or POS.

    Returns:
        Processed results DataFrame.
    """
    dataset_product = dataset.loc[dataset["PRODUCT"] == product].copy()

    marginal_column = build_marginal_column(product_sign=product_sign)
    if marginal_column not in dataset_product.columns:
        raise KeyError(f"Missing marginal column: {marginal_column}")

    if "ACTUAL_VALUE" not in results.columns:
        raise KeyError("ACTUAL_VALUE column not found in results file.")

    marginal_series = dataset_product[marginal_column].reindex(results.index)

    processed = results.copy()
    processed["MARGINAL_PRICE"] = marginal_series

    processed = processed.loc[
        processed["ACTUAL_VALUE"].notna() &
        processed["D+1"].notna()
    ].copy()

    processed["RMSE_COMPONENT"] = (processed["D+1"] - processed["ACTUAL_VALUE"]) ** 2
    processed["MAE"] = np.abs(processed["D+1"] - processed["ACTUAL_VALUE"])
    processed["MAPE"] = np.where(
        processed["ACTUAL_VALUE"] != 0,
        np.abs((processed["D+1"] - processed["ACTUAL_VALUE"]) / processed["ACTUAL_VALUE"]),
        np.nan,
    )
    processed["ACCEPTANCE"] = np.where(
        processed["MARGINAL_PRICE"].notna() &
        (processed["D+1"] <= processed["MARGINAL_PRICE"]),
        1,
        0,
    )
    processed["REVENUE"] = np.where(processed["ACCEPTANCE"] == 1, processed["D+1"], 0.0)

    extra_prediction_cols = [
        col for col in processed.columns
        if col.startswith("D+1_") and col != "D+1"
    ]
    if extra_prediction_cols:
        processed = processed.drop(columns=extra_prediction_cols)

    return processed


def save_processed_results(
    processed: pd.DataFrame,
    postprocessed_root: Path,
    model_key: str,
    product: str,
    product_sign: str,
    price_type: str,
    lookback: Optional[int] = None,
    strategy: Optional[str] = None,
) -> Path:
    """
    Save one processed result file.

    Parameters:
        processed: Processed DataFrame.
        postprocessed_root: Root postprocessed folder.
        model_key: Internal model key.
        product: Product block.
        product_sign: NEG or POS.
        price_type: AVERAGE or MARGINAL.
        lookback: Optional lookback.
        strategy: Optional strategy.

    Returns:
        Saved file path.
    """
    model_dir = postprocessed_root / "processed_results" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    filename_parts = [model_key]
    if lookback is not None:
        filename_parts.append(f"ts{lookback}")
    if strategy is not None:
        filename_parts.append(strategy)

    filename_parts.extend(["processed_results", product, product_sign, price_type])

    output_file = model_dir / ("_".join(filename_parts) + ".csv")
    processed.to_csv(output_file, index=True)

    return output_file


def build_model_name(
    model_key: str,
    lookback: Optional[int] = None,
    strategy: Optional[str] = None,
) -> str:
    """
    Build internal MODEL value for summary tables.

    Parameters:
        model_key: Internal model key.
        lookback: Optional lookback.
        strategy: Optional strategy.

    Returns:
        Internal model string.
    """
    name = model_key
    if lookback is not None:
        name += f"_ts{lookback}"
    if strategy is not None:
        name += f"_{strategy}"
    return name


def build_pretty_model_name(
    model_key: str,
    lookback: Optional[int] = None,
    strategy: Optional[str] = None,
) -> str:
    """
    Build human-readable model label.

    Parameters:
        model_key: Internal model key.
        lookback: Optional lookback.
        strategy: Optional strategy.

    Returns:
        Pretty model label.
    """
    base_names = {
        "arma": "ARMA",
        "armax_exog_1": "ARMAX (Exog. 1)",
        "armax_exog_4": "ARMAX (Exog. 4)",
        "armax_exog_1_3": "ARMAX (Exog. 1&3)",
        "armax_exog_1_4": "ARMAX (Exog. 1&4)",
        "sarma": "SARMA",
        "sarmax_exog_1": "SARMAX (Exog. 1)",
        "sarmax_exog_4": "SARMAX (Exog. 4)",
        "sarmax_exog_1_3": "SARMAX (Exog. 1&3)",
        "sarmax_exog_1_4": "SARMAX (Exog. 1&4)",

        "rw_arma_3m": "RW-ARMA (3 Months)",
        "rw_arma_6m": "RW-ARMA (6 Months)",
        "rw_arma_12m": "RW-ARMA (12 Months)",

        "rw_armax_exog_1_3m": "RW-ARMAX (Exog. 1) (3 Months)",
        "rw_armax_exog_1_6m": "RW-ARMAX (Exog. 1) (6 Months)",
        "rw_armax_exog_1_12m": "RW-ARMAX (Exog. 1) (12 Months)",

        "rw_armax_exog_4_3m": "RW-ARMAX (Exog. 4) (3 Months)",
        "rw_armax_exog_4_6m": "RW-ARMAX (Exog. 4) (6 Months)",
        "rw_armax_exog_4_12m": "RW-ARMAX (Exog. 4) (12 Months)",

        "rw_armax_exog_1_3_3m": "RW-ARMAX (Exog. 1&3) (3 Months)",
        "rw_armax_exog_1_3_6m": "RW-ARMAX (Exog. 1&3) (6 Months)",
        "rw_armax_exog_1_3_12m": "RW-ARMAX (Exog. 1&3) (12 Months)",

        "rw_armax_exog_1_4_3m": "RW-ARMAX (Exog. 1&4) (3 Months)",
        "rw_armax_exog_1_4_6m": "RW-ARMAX (Exog. 1&4) (6 Months)",
        "rw_armax_exog_1_4_12m": "RW-ARMAX (Exog. 1&4) (12 Months)",

        "rw_sarma_3m": "RW-SARMA (3 Months)",
        "rw_sarma_6m": "RW-SARMA (6 Months)",
        "rw_sarma_12m": "RW-SARMA (12 Months)",

        "rw_sarmax_exog_1_3m": "RW-SARMAX (Exog. 1) (3 Months)",
        "rw_sarmax_exog_1_6m": "RW-SARMAX (Exog. 1) (6 Months)",
        "rw_sarmax_exog_1_12m": "RW-SARMAX (Exog. 1) (12 Months)",

        "rw_sarmax_exog_4_3m": "RW-SARMAX (Exog. 4) (3 Months)",
        "rw_sarmax_exog_4_6m": "RW-SARMAX (Exog. 4) (6 Months)",
        "rw_sarmax_exog_4_12m": "RW-SARMAX (Exog. 4) (12 Months)",

        "rw_sarmax_exog_1_3_3m": "RW-SARMAX (Exog. 1&3) (3 Months)",
        "rw_sarmax_exog_1_3_6m": "RW-SARMAX (Exog. 1&3) (6 Months)",
        "rw_sarmax_exog_1_3_12m": "RW-SARMAX (Exog. 1&3) (12 Months)",

        "rw_sarmax_exog_1_4_3m": "RW-SARMAX (Exog. 1&4) (3 Months)",
        "rw_sarmax_exog_1_4_6m": "RW-SARMAX (Exog. 1&4) (6 Months)",
        "rw_sarmax_exog_1_4_12m": "RW-SARMAX (Exog. 1&4) (12 Months)",

        "lstm": "LSTM",
        "bidirectional_lstm": "BiLSTM",
        "lstm_features_1": "LSTM with Feature 1",
        "lstm_features_4": "LSTM with Feature 4",
        "lstm_features_1_3": "LSTM with Features 1&3",
        "lstm_features_1_4": "LSTM with Features 1&4",

        "xgboost_exog_1": "XGBoost (Exog. 1)",
        "xgboost_exog_4": "XGBoost (Exog. 4)",
        "xgboost_exog_1_3": "XGBoost (Exog. 1&3)",
        "xgboost_exog_1_4": "XGBoost (Exog. 1&4)",

        "xgboost_v2_exog_1": "XGBoost v2 (Exog. 1)",
        "xgboost_v2_exog_4": "XGBoost v2 (Exog. 4)",
        "xgboost_v2_exog_1_3": "XGBoost v2 (Exog. 1&3)",
        "xgboost_v2_exog_1_4": "XGBoost v2 (Exog. 1&4)",
    }

    strategy_names = {
        "single": "Single",
        "ensemble": "Ensemble",
        "recency": "Weighted Ensemble",
    }

    name = base_names.get(model_key, model_key)

    if lookback is not None:
        name = f"{name} ({lookback} TimeSteps)"
    if strategy is not None:
        name = f"{name}, {strategy_names.get(strategy, strategy)}"

    return name


def summarize_processed_results(
    processed: pd.DataFrame,
    model_key: str,
    product: str,
    product_sign: str,
    price_type: str,
    model_group: str,
    lookback: Optional[int] = None,
    strategy: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build one-row summary metrics DataFrame.

    Parameters:
        processed: Processed results DataFrame.
        model_key: Internal model key.
        product: Product block.
        product_sign: NEG or POS.
        price_type: AVERAGE or MARGINAL.
        model_group: High-level group.
        lookback: Optional lookback.
        strategy: Optional strategy.

    Returns:
        Single-row DataFrame.
    """
    actual_sum = float(processed["ACTUAL_VALUE"].sum()) if not processed.empty else np.nan
    revenue_sum = float(processed["REVENUE"].sum()) if not processed.empty else np.nan

    return pd.DataFrame([{
        "MODEL": build_model_name(model_key=model_key, lookback=lookback, strategy=strategy),
        "MODEL_BASE": model_key,
        "MODEL_NAME": build_pretty_model_name(model_key=model_key, lookback=lookback, strategy=strategy),
        "MODEL_GROUP": model_group,
        "LOOKBACK": lookback,
        "STRATEGY": strategy,
        "PRODUCT": product,
        "PRODUCT_SIGN": product_sign,
        "PRICE_TYPE": price_type,
        "N_OBSERVATIONS": int(len(processed)),
        "RMSE": float(np.sqrt(processed["RMSE_COMPONENT"].mean())) if len(processed) > 0 else np.nan,
        "MAPE": float(processed["MAPE"].mean()) if len(processed) > 0 else np.nan,
        "MAE": float(processed["MAE"].mean()) if len(processed) > 0 else np.nan,
        "MED_MAE": float(processed["MAE"].median()) if len(processed) > 0 else np.nan,
        "ZERO_COUNT": int((processed["D+1"].isna() | (processed["D+1"] == 0)).sum()),
        "NEGATIVE_COUNT": int((processed["D+1"] < 0).sum()),
        "ACCEPTANCE": int(processed["ACCEPTANCE"].sum()) if len(processed) > 0 else 0,
        "CHANCE_OF_ACCEPTANCE": float(processed["ACCEPTANCE"].mean()) if len(processed) > 0 else np.nan,
        "REVENUE": revenue_sum,
        "MAX_REVENUE": actual_sum,
        "ACHIEVED_REVENUE_PERCENTAGE": (
            (revenue_sum / actual_sum) * 100.0
            if pd.notna(revenue_sum) and pd.notna(actual_sum) and actual_sum != 0
            else np.nan
        ),
    }])


def process_one_configuration(
    dataset: pd.DataFrame,
    base_output_path: Path,
    postprocessed_root: Path,
    spec: ModelSpec,
    product: str,
    product_sign: str,
    price_type: str,
    lookback: Optional[int] = None,
    strategy: Optional[str] = None,
    min_date: str = "2024-07-01",
) -> Optional[pd.DataFrame]:
    """
    Process one model/product/sign/type configuration.

    Parameters:
        dataset: Main processed dataset.
        base_output_path: Base model output path.
        postprocessed_root: Root folder for postprocessed files.
        spec: Model specification.
        product: Product block.
        product_sign: NEG or POS.
        price_type: AVERAGE or MARGINAL.
        lookback: Optional lookback.
        strategy: Optional LSTM strategy.
        min_date: Minimum included date.

    Returns:
        Single-row summary DataFrame, or None if file is missing or invalid.
    """
    results_file = get_result_file_path(
        base_output_path=base_output_path,
        spec=spec,
        product=product,
        product_sign=product_sign,
        price_type=price_type,
        lookback=lookback,
    )

    if "xgboost" in spec.model_key:
        print(f"CHECKING: {spec.model_key} -> {results_file} | exists={results_file.exists()}")

    if not results_file.exists():
        LOGGER.warning("Missing result file: %s", results_file)
        return None

    try:
        results = load_results_file(results_file)
        results = results.loc[results.index >= pd.Timestamp(min_date)].copy()

        if results.empty:
            LOGGER.warning("Empty results after date filter: %s", results_file)
            return None

        results = select_prediction_column(
            results=results,
            model_key=spec.model_key,
            strategy=strategy,
        )

        processed = compute_processed_results(
            dataset=dataset,
            results=results,
            product=product,
            product_sign=product_sign,
        )

        if processed.empty:
            LOGGER.warning("Processed results are empty: %s", results_file)
            return None

        save_processed_results(
            processed=processed,
            postprocessed_root=postprocessed_root,
            model_key=spec.model_key,
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            lookback=lookback,
            strategy=strategy,
        )

        return summarize_processed_results(
            processed=processed,
            model_key=spec.model_key,
            product=product,
            product_sign=product_sign,
            price_type=price_type,
            model_group=spec.group,
            lookback=lookback,
            strategy=strategy,
        )

    except Exception as exc:
        LOGGER.exception(
            "Failed to process spec=%s, product=%s, sign=%s, type=%s, lookback=%s, strategy=%s: %s",
            spec.model_key,
            product,
            product_sign,
            price_type,
            lookback,
            strategy,
            exc,
        )
        return None


def add_scores_and_post_metrics(all_models_comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized scores, revenue scaling, and MAE_vs_BEST_STATISTICAL.

    Parameters:
        all_models_comparison: Master comparison DataFrame.

    Returns:
        Enriched DataFrame.
    """
    df = all_models_comparison.copy()

    if df.empty:
        return df

    df["MAPE_NORM"] = robust_normalize(df["MAPE"], upper_percentile=99.0)
    df["RMSE_NORM"] = robust_normalize(df["RMSE"], upper_percentile=99.0)
    df["MAE_NORM"] = robust_normalize(df["MAE"], upper_percentile=99.0)

    df["COMBINED_SCORE"] = (
        0.34 * df["MAPE_NORM"] +
        0.33 * df["RMSE_NORM"] +
        0.33 * df["MAE_NORM"]
    )

    df["REVENUE"] = df["REVENUE"] * 0.004
    df["MAX_REVENUE"] = df["MAX_REVENUE"] * 0.004

    df["MAE_vs_BEST_STATISTICAL"] = np.nan

    statistical_groups = {"STATISTICAL_EXPANDING", "STATISTICAL_ROLLING"}

    for product in sorted(df["PRODUCT"].dropna().unique()):
        for product_sign in sorted(df["PRODUCT_SIGN"].dropna().unique()):
            for price_type in sorted(df["PRICE_TYPE"].dropna().unique()):
                mask = (
                    (df["PRODUCT"] == product) &
                    (df["PRODUCT_SIGN"] == product_sign) &
                    (df["PRICE_TYPE"] == price_type)
                )

                statistical_subset = df.loc[
                    mask & df["MODEL_GROUP"].isin(statistical_groups)
                ].copy()

                if statistical_subset.empty:
                    continue

                best_idx = statistical_subset["COMBINED_SCORE"].idxmax()
                best_med_mae = df.loc[best_idx, "MED_MAE"]

                if pd.isna(best_med_mae) or best_med_mae == 0:
                    continue

                df.loc[mask, "MAE_vs_BEST_STATISTICAL"] = (
                    (df.loc[mask, "MED_MAE"] - best_med_mae) / best_med_mae
                ) * 100.0

    return df


def export_group_splits(
    all_models_comparison: pd.DataFrame,
    output_folder: Path,
) -> None:
    """
    Export per-group filtered comparison tables.

    Parameters:
        all_models_comparison: Master comparison DataFrame.
        output_folder: Postprocessed output folder.

    Returns:
        None
    """
    if all_models_comparison.empty:
        return

    for group_name in sorted(all_models_comparison["MODEL_GROUP"].dropna().unique()):
        group_df = all_models_comparison.loc[
            all_models_comparison["MODEL_GROUP"] == group_name
        ].copy()

        output_file = output_folder / f"{group_name.lower()}_models_comparison.csv"
        group_df.to_csv(output_file, index=False)
        LOGGER.info("Saved group comparison file: %s", output_file)


def export_best_model_tables(
    all_models_comparison: pd.DataFrame,
    output_folder: Path,
) -> None:
    """
    Export best-model tables by criteria and group.

    Parameters:
        all_models_comparison: Master comparison DataFrame.
        output_folder: Output folder.

    Returns:
        None
    """
    if all_models_comparison.empty:
        return

    group_cols = ["PRODUCT", "PRODUCT_SIGN", "PRICE_TYPE"]

    best_errors_combined = (
        all_models_comparison.sort_values("COMBINED_SCORE", ascending=False)
        .groupby(group_cols, dropna=False)
        .first()
    )
    best_revenue = (
        all_models_comparison.sort_values("REVENUE", ascending=False)
        .groupby(group_cols, dropna=False)
        .first()
    )

    best_errors_combined.to_csv(output_folder / "best_models_by_combined_score_of_errors.csv")
    best_revenue.to_csv(output_folder / "best_models_by_revenue.csv")

    grouped_exports = {
        "best_statistical_models_by_combined_score.csv": {"STATISTICAL_EXPANDING", "STATISTICAL_ROLLING"},
        "best_statistical_models_by_revenue.csv": {"STATISTICAL_EXPANDING", "STATISTICAL_ROLLING"},
        "best_deep_learning_models_by_combined_score.csv": {"DEEP_LEARNING"},
        "best_deep_learning_models_by_revenue.csv": {"DEEP_LEARNING"},
        "best_tree_based_models_by_combined_score.csv": {"TREE_BASED"},
        "best_tree_based_models_by_revenue.csv": {"TREE_BASED"},
    }

    for filename, allowed_groups in grouped_exports.items():
        subset = all_models_comparison.loc[
            all_models_comparison["MODEL_GROUP"].isin(allowed_groups)
        ].copy()

        if subset.empty:
            continue

        if "combined_score" in filename:
            best_subset = (
                subset.sort_values("COMBINED_SCORE", ascending=False)
                .groupby(group_cols, dropna=False)
                .first()
            )
        else:
            best_subset = (
                subset.sort_values("REVENUE", ascending=False)
                .groupby(group_cols, dropna=False)
                .first()
            )

        best_subset.to_csv(output_folder / filename)
        LOGGER.info("Saved best-model table: %s", output_folder / filename)


def build_model_registry() -> list[ModelSpec]:
    """
    Define all model families to be postprocessed.

    Returns:
        List of model specifications.
    """
    return [
        ModelSpec("arma", "arma_model_output", "arma", "STATISTICAL_EXPANDING"),
        ModelSpec("armax_exog_1", "armax_exog_1_model_output", "armax_exog_1", "STATISTICAL_EXPANDING"),
        ModelSpec("armax_exog_4", "armax_exog_4_model_output", "armax_exog_4", "STATISTICAL_EXPANDING"),
        ModelSpec("armax_exog_1_3", "armax_exog_1_3_model_output", "armax_exog_1_3", "STATISTICAL_EXPANDING"),
        ModelSpec("armax_exog_1_4", "armax_exog_1_4_model_output", "armax_exog_1_4", "STATISTICAL_EXPANDING"),
        ModelSpec("sarma", "sarma_model_output", "sarma", "STATISTICAL_EXPANDING"),
        ModelSpec("sarmax_exog_1", "sarmax_exog_1_model_output", "sarmax_exog_1", "STATISTICAL_EXPANDING"),
        ModelSpec("sarmax_exog_4", "sarmax_exog_4_model_output", "sarmax_exog_4", "STATISTICAL_EXPANDING"),
        ModelSpec("sarmax_exog_1_3", "sarmax_exog_1_3_model_output", "sarmax_exog_1_3", "STATISTICAL_EXPANDING"),
        ModelSpec("sarmax_exog_1_4", "sarmax_exog_1_4_model_output", "sarmax_exog_1_4", "STATISTICAL_EXPANDING"),

        ModelSpec("rw_arma_3m", "rw_arma_3m_model_output", "rw_arma_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_arma_6m", "rw_arma_6m_model_output", "rw_arma_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_arma_12m", "rw_arma_12m_model_output", "rw_arma_12m", "STATISTICAL_ROLLING"),

        ModelSpec("rw_armax_exog_1_3m", "rw_armax_exog_1_3m_model_output", "rw_armax_exog_1_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_armax_exog_1_6m", "rw_armax_exog_1_6m_model_output", "rw_armax_exog_1_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_armax_exog_1_12m", "rw_armax_exog_1_12m_model_output", "rw_armax_exog_1_12m", "STATISTICAL_ROLLING"),

        ModelSpec("rw_armax_exog_4_3m", "rw_armax_exog_4_3m_model_output", "rw_armax_exog_4_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_armax_exog_4_6m", "rw_armax_exog_4_6m_model_output", "rw_armax_exog_4_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_armax_exog_4_12m", "rw_armax_exog_4_12m_model_output", "rw_armax_exog_4_12m", "STATISTICAL_ROLLING"),

        ModelSpec("rw_armax_exog_1_3_3m", "rw_armax_exog_1_3_3m_model_output", "rw_armax_exog_1_3_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_armax_exog_1_3_6m", "rw_armax_exog_1_3_6m_model_output", "rw_armax_exog_1_3_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_armax_exog_1_3_12m", "rw_armax_exog_1_3_12m_model_output", "rw_armax_exog_1_3_12m", "STATISTICAL_ROLLING"),

        ModelSpec("rw_armax_exog_1_4_3m", "rw_armax_exog_1_4_3m_model_output", "rw_armax_exog_1_4_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_armax_exog_1_4_6m", "rw_armax_exog_1_4_6m_model_output", "rw_armax_exog_1_4_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_armax_exog_1_4_12m", "rw_armax_exog_1_4_12m_model_output", "rw_armax_exog_1_4_12m", "STATISTICAL_ROLLING"),

        ModelSpec("rw_sarma_3m", "rw_sarma_3m_model_output", "rw_sarma_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarma_6m", "rw_sarma_6m_model_output", "rw_sarma_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarma_12m", "rw_sarma_12m_model_output", "rw_sarma_12m", "STATISTICAL_ROLLING"),

        ModelSpec("rw_sarmax_exog_1_3m", "rw_sarmax_exog_1_3m_model_output", "rw_sarmax_exog_1_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarmax_exog_1_6m", "rw_sarmax_exog_1_6m_model_output", "rw_sarmax_exog_1_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarmax_exog_1_12m", "rw_sarmax_exog_1_12m_model_output", "rw_sarmax_exog_1_12m", "STATISTICAL_ROLLING"),

        ModelSpec("rw_sarmax_exog_4_3m", "rw_sarmax_exog_4_3m_model_output", "rw_sarmax_exog_4_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarmax_exog_4_6m", "rw_sarmax_exog_4_6m_model_output", "rw_sarmax_exog_4_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarmax_exog_4_12m", "rw_sarmax_exog_4_12m_model_output", "rw_sarmax_exog_4_12m", "STATISTICAL_ROLLING"),

        ModelSpec("rw_sarmax_exog_1_3_3m", "rw_sarmax_exog_1_3_3m_model_output", "rw_sarmax_exog_1_3_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarmax_exog_1_3_6m", "rw_sarmax_exog_1_3_6m_model_output", "rw_sarmax_exog_1_3_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarmax_exog_1_3_12m", "rw_sarmax_exog_1_3_12m_model_output", "rw_sarmax_exog_1_3_12m", "STATISTICAL_ROLLING"),

        ModelSpec("rw_sarmax_exog_1_4_3m", "rw_sarmax_exog_1_4_3m_model_output", "rw_sarmax_exog_1_4_3m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarmax_exog_1_4_6m", "rw_sarmax_exog_1_4_6m_model_output", "rw_sarmax_exog_1_4_6m", "STATISTICAL_ROLLING"),
        ModelSpec("rw_sarmax_exog_1_4_12m", "rw_sarmax_exog_1_4_12m_model_output", "rw_sarmax_exog_1_4_12m", "STATISTICAL_ROLLING"),

        ModelSpec(
            "lstm",
            "lstm_model_output",
            "lstm",
            "DEEP_LEARNING",
            lookbacks=[42, 168],
            strategies=["ensemble", "recency", "single"],
        ),
        ModelSpec(
            "bidirectional_lstm",
            "bidirectional_lstm_model_output",
            "bidirectional_lstm",
            "DEEP_LEARNING",
            lookbacks=[42, 168],
            strategies=["ensemble", "recency", "single"],
        ),
        ModelSpec(
            "lstm_features_1",
            "lstm_features_1_model_output",
            "lstm_features_1",
            "DEEP_LEARNING",
            lookbacks=[42, 168],
            strategies=["ensemble", "recency", "single"],
        ),
        ModelSpec(
            "lstm_features_4",
            "lstm_features_4_model_output",
            "lstm_features_4",
            "DEEP_LEARNING",
            lookbacks=[42, 168],
            strategies=["ensemble", "recency", "single"],
        ),
        ModelSpec(
            "lstm_features_1_3",
            "lstm_features_1_3_model_output",
            "lstm_features_1_3",
            "DEEP_LEARNING",
            lookbacks=[42, 168],
            strategies=["ensemble", "recency", "single"],
        ),
        ModelSpec(
            "lstm_features_1_4",
            "lstm_features_1_4_model_output",
            "lstm_features_1_4",
            "DEEP_LEARNING",
            lookbacks=[42, 168],
            strategies=["ensemble", "recency", "single"],
        ),

        ModelSpec("xgboost_exog_4", "xgboost_exog_4_model_output", "xgboost_exog_4", "TREE_BASED"),

        ModelSpec(
            "xgboost_v2_exog_1",
            "xgboost_v2_exog_1_model_output",
            "xgboost_v2_exog_1",
            "TREE_BASED",
            lookbacks=[7],
        ),
        ModelSpec(
            "xgboost_v2_exog_4",
            "xgboost_v2_exog_4_model_output",
            "xgboost_v2_exog_4",
            "TREE_BASED",
            lookbacks=[7],
        ),
        ModelSpec(
            "xgboost_v2_exog_1_3",
            "xgboost_v2_exog_1_3_model_output",
            "xgboost_v2_exog_1_3",
            "TREE_BASED",
            lookbacks=[7],
        ),
        ModelSpec(
            "xgboost_v2_exog_1_4",
            "xgboost_v2_exog_1_4_model_output",
            "xgboost_v2_exog_1_4",
            "TREE_BASED",
            lookbacks=[7],
        ),
    ]


def check_observation_consistency(all_models_comparison: pd.DataFrame) -> None:
    """
    Check whether all models use the same number of observations
    for a fixed configuration.

    Prints a clear confirmation or warning.

    Parameters:
        all_models_comparison: Master comparison DataFrame

    Returns:
        None
    """
    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK: N_OBSERVATIONS")
    print("=" * 80)

    subset = all_models_comparison.loc[
        (all_models_comparison["PRODUCT"] == "00_04") &
        (all_models_comparison["PRODUCT_SIGN"] == "NEG") &
        (all_models_comparison["PRICE_TYPE"] == "AVERAGE")
    ].copy()

    if subset.empty:
        print("⚠️ No data found for consistency check.")
        return

    subset = subset.sort_values(["MODEL_GROUP", "MODEL"])

    print("\nSubset used for check (00_04 / NEG / AVERAGE):\n")
    print(
        subset[
            ["MODEL", "MODEL_BASE", "MODEL_GROUP", "LOOKBACK", "N_OBSERVATIONS"]
        ].to_string(index=False)
    )

    unique_counts = subset["N_OBSERVATIONS"].dropna().unique()

    print("\n--- RESULT ---")

    if len(unique_counts) == 1:
        print(f"✅ All models use the SAME number of observations: {unique_counts[0]}")
    else:
        print("⚠️ MISMATCH detected in N_OBSERVATIONS!")
        print(f"Different values found: {sorted(unique_counts)}")

        print("\nDetailed differences:")
        print(
            subset[
                ["MODEL", "MODEL_GROUP", "LOOKBACK", "N_OBSERVATIONS"]
            ]
            .sort_values("N_OBSERVATIONS")
            .to_string(index=False)
        )

    print("=" * 80 + "\n")


def check_date_alignment(all_models_comparison: pd.DataFrame) -> None:
    """
    Check whether all models share the exact same DATE range
    for a fixed configuration.

    Prints min/max DATE per model and flags mismatches.

    Parameters:
        all_models_comparison: Master comparison DataFrame

    Returns:
        None
    """
    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK: DATE ALIGNMENT")
    print("=" * 80)

    subset = all_models_comparison.loc[
        (all_models_comparison["PRODUCT"] == "00_04") &
        (all_models_comparison["PRODUCT_SIGN"] == "NEG") &
        (all_models_comparison["PRICE_TYPE"] == "AVERAGE")
    ].copy()

    if subset.empty:
        print("⚠️ No data found for DATE alignment check.")
        return

    # We must reload processed files to inspect actual DATE index
    from paths import BASE_OUTPUT_PATH
    postprocessed_root = BASE_OUTPUT_PATH / "postprocessed_data" / "processed_results"

    rows = []

    for _, row in subset.iterrows():
        model_key = row["MODEL_BASE"]
        model_name = row["MODEL"]

        model_dir = postprocessed_root / model_key

        # find matching file (since naming includes product etc.)
        matches = list(model_dir.glob(f"{model_name}_processed_results_00_04_NEG_AVERAGE.csv"))

        if not matches:
            continue

        file_path = matches[0]

        df = pd.read_csv(file_path, parse_dates=["DATE"], index_col="DATE")

        if df.empty:
            continue

        rows.append({
            "MODEL": model_name,
            "MODEL_GROUP": row["MODEL_GROUP"],
            "MIN_DATE": df.index.min(),
            "MAX_DATE": df.index.max(),
            "N_ROWS": len(df),
        })

    if not rows:
        print("⚠️ No processed files found for DATE check.")
        return

    df_check = pd.DataFrame(rows).sort_values("MODEL")

    print("\nDATE ranges per model:\n")
    print(df_check.to_string(index=False))

    min_dates = df_check["MIN_DATE"].unique()
    max_dates = df_check["MAX_DATE"].unique()

    print("\n--- RESULT ---")

    if len(min_dates) == 1 and len(max_dates) == 1:
        print(f"✅ All models share identical DATE range: {min_dates[0]} → {max_dates[0]}")
    else:
        print("⚠️ DATE MISALIGNMENT detected!")

        print("\nDifferent MIN_DATE values:")
        print(sorted(min_dates))

        print("\nDifferent MAX_DATE values:")
        print(sorted(max_dates))

        print("\nModels with differences:")
        print(df_check.to_string(index=False))

    print("=" * 80 + "\n")

def main() -> None:
    """
    Main entry point.

    Parameters:
        None

    Returns:
        None
    """
    configure_logging()

    dataset_path = BASE_INPUT_PATH / "processed_afrr_data.csv"
    output_folder = BASE_OUTPUT_PATH / "postprocessed_data"
    output_folder.mkdir(parents=True, exist_ok=True)

    processed_afrr_data = load_dataset(dataset_path)
    processed_afrr_data = processed_afrr_data.loc[
        processed_afrr_data.index >= pd.Timestamp("2024-07-01")
    ].copy()

    products = ["00_04", "04_08", "08_12", "12_16", "16_20", "20_24"]
    product_signs = ["NEG", "POS"]
    price_types = ["AVERAGE", "MARGINAL"]

    model_specs = build_model_registry()
    summary_rows: list[pd.DataFrame] = []

    for spec in model_specs:
        lookbacks = spec.lookbacks if spec.lookbacks is not None else [None]
        strategies = spec.strategies if spec.strategies is not None else [None]

        for lookback in lookbacks:
            for strategy in strategies:
                for product in products:
                    for product_sign in product_signs:
                        for price_type in price_types:
                            LOGGER.info(
                                "Processing model=%s, lookback=%s, strategy=%s, product=%s, sign=%s, type=%s",
                                spec.model_key,
                                lookback,
                                strategy,
                                product,
                                product_sign,
                                price_type,
                            )

                            postprocessed_df = process_one_configuration(
                                dataset=processed_afrr_data,
                                base_output_path=BASE_OUTPUT_PATH,
                                postprocessed_root=output_folder,
                                spec=spec,
                                product=product,
                                product_sign=product_sign,
                                price_type=price_type,
                                lookback=lookback,
                                strategy=strategy,
                                min_date="2024-07-01",
                            )

                            if postprocessed_df is not None:
                                summary_rows.append(postprocessed_df)

    if summary_rows:
        all_models_comparison = pd.concat(summary_rows, ignore_index=True)
    else:
        all_models_comparison = pd.DataFrame()

    if all_models_comparison.empty:
        LOGGER.warning("No model results were successfully postprocessed.")
        all_models_comparison.to_csv(output_folder / "all_models_comparison.csv", index=False)
        return

    all_models_comparison = add_scores_and_post_metrics(all_models_comparison)
    all_models_comparison = all_models_comparison.sort_values(
        by=["MODEL_GROUP", "MODEL", "PRODUCT", "PRODUCT_SIGN", "PRICE_TYPE"],
        ignore_index=True,
    )

    all_models_comparison.to_csv(output_folder / "all_models_comparison.csv", index=False)
    LOGGER.info("Saved master comparison table.")

    export_group_splits(
        all_models_comparison=all_models_comparison,
        output_folder=output_folder,
    )

    export_best_model_tables(
        all_models_comparison=all_models_comparison,
        output_folder=output_folder,
    )

    print("\nXGBoost rows in all_models_comparison:")
    print(
        all_models_comparison.loc[
            all_models_comparison["MODEL_BASE"].str.contains("xgboost", case=False, na=False),
            [
                "MODEL",
                "MODEL_BASE",
                "PRODUCT",
                "PRODUCT_SIGN",
                "PRICE_TYPE",
                "LOOKBACK",
                "N_OBSERVATIONS",
                "RMSE",
                "MAPE",
                "MAE",
                "MED_MAE",
                "REVENUE",
                "COMBINED_SCORE",
            ],
        ].to_string(index=False)
    )

    LOGGER.info("Postprocessing completed successfully.")
    # Consistency check (important for fair comparison)
    check_observation_consistency(all_models_comparison)
    check_date_alignment(all_models_comparison)


# Example usage
# python3 -m postprocessing.results_postprocessing
if __name__ == "__main__":
    main()