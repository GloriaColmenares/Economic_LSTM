# 01_Python_code/postprocessing/model_diagnostics.py

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

# --------------------------------------------------------------------------------------
# Import project paths safely
# --------------------------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
PROJECT_CODE_ROOT = CURRENT_FILE.parents[1]  # 01_Python_code
if str(PROJECT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_ROOT))

from paths import BASE_OUTPUT_PATH  # noqa: E402


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
class DiagnosticConfig:
    """
    Stores configuration for diagnostics.

    Parameters:
        postprocessed_root: Root folder with postprocessed outputs.
        diagnostics_root: Output folder for diagnostic tables.
        min_date: Minimum date already used in postprocessing.
        spike_quantile: Quantile used to define spikes.
        n_level_bins: Number of ACTUAL_VALUE bins for level analysis.
        focus_product: Product used for consistency/spot checks.
        focus_product_sign: Product sign used for consistency/spot checks.
        focus_price_type: Price type used for consistency/spot checks.
    """

    postprocessed_root: Path
    diagnostics_root: Path
    min_date: str = "2024-07-01"
    spike_quantile: float = 0.95
    n_level_bins: int = 5
    focus_product: str = "00_04"
    focus_product_sign: str = "NEG"
    focus_price_type: str = "AVERAGE"


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def load_master_comparison(postprocessed_root: Path) -> pd.DataFrame:
    """
    Load the master comparison table.

    Parameters:
        postprocessed_root: Postprocessed data folder.

    Returns:
        Master comparison DataFrame.
    """
    file_path = postprocessed_root / "all_models_comparison.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)

    if "LOOKBACK" in df.columns:
        df["LOOKBACK"] = pd.to_numeric(df["LOOKBACK"], errors="coerce")
    if "N_OBSERVATIONS" in df.columns:
        df["N_OBSERVATIONS"] = pd.to_numeric(df["N_OBSERVATIONS"], errors="coerce")

    return df


def load_processed_result_file(file_path: Path) -> pd.DataFrame:
    """
    Load one processed results file.

    Parameters:
        file_path: Path to processed result CSV.

    Returns:
        Loaded DataFrame with DATE index.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Missing processed file: {file_path}")

    df = pd.read_csv(file_path)

    if "DATE" not in df.columns:
        raise KeyError(f"'DATE' column not found in {file_path}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.loc[df["DATE"].notna()].copy()
    df = df.set_index("DATE").sort_index()

    return df


def build_processed_result_path(
    postprocessed_root: Path,
    model_base: str,
    product: str,
    product_sign: str,
    price_type: str,
    lookback: Optional[int],
    strategy: Optional[str],
) -> Path:
    """
    Build processed result file path from model metadata.

    Parameters:
        postprocessed_root: Root postprocessed folder.
        model_base: Base model key.
        product: Product block.
        product_sign: Regulation sign.
        price_type: Price type.
        lookback: Optional lookback.
        strategy: Optional prediction strategy.

    Returns:
        Expected processed result file path.
    """
    parts: list[str] = [model_base]

    if lookback is not None and not pd.isna(lookback):
        parts.append(f"ts{int(lookback)}")

    if strategy is not None and not pd.isna(strategy):
        parts.append(str(strategy))

    parts.extend(["processed_results", product, product_sign, price_type])

    filename = "_".join(parts) + ".csv"
    return postprocessed_root / "processed_results" / model_base / filename


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add error-related columns.

    Parameters:
        df: Processed result DataFrame.

    Returns:
        DataFrame with additional error columns.
    """
    out = df.copy()

    required_cols = {"ACTUAL_VALUE", "D+1"}
    missing_cols = required_cols.difference(out.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {sorted(missing_cols)}")

    out["ERROR"] = out["D+1"] - out["ACTUAL_VALUE"]
    out["ABS_ERROR"] = np.abs(out["ERROR"])
    out["SQUARED_ERROR"] = out["ERROR"] ** 2
    out["PRED_CHANGE"] = out["D+1"].diff()
    out["ACTUAL_CHANGE"] = out["ACTUAL_VALUE"].diff()

    return out


def compute_bias_variance_metrics(
    df: pd.DataFrame,
    model_row: pd.Series,
) -> dict:
    """
    Compute bias/variance style metrics.

    Parameters:
        df: Processed result DataFrame with error columns.
        model_row: Row from all_models_comparison.

    Returns:
        Dictionary of metrics.
    """
    error = df["ERROR"].dropna()

    return {
        "MODEL": model_row["MODEL"],
        "MODEL_BASE": model_row["MODEL_BASE"],
        "MODEL_NAME": model_row["MODEL_NAME"],
        "MODEL_GROUP": model_row["MODEL_GROUP"],
        "PRODUCT": model_row["PRODUCT"],
        "PRODUCT_SIGN": model_row["PRODUCT_SIGN"],
        "PRICE_TYPE": model_row["PRICE_TYPE"],
        "LOOKBACK": model_row["LOOKBACK"],
        "STRATEGY": model_row["STRATEGY"],
        "N_OBSERVATIONS": int(len(df)),
        "MEAN_ERROR_BIAS": float(error.mean()) if not error.empty else np.nan,
        "MEDIAN_ERROR_BIAS": float(error.median()) if not error.empty else np.nan,
        "ERROR_STD": float(error.std(ddof=1)) if len(error) > 1 else np.nan,
        "ERROR_VAR": float(error.var(ddof=1)) if len(error) > 1 else np.nan,
        "ABS_ERROR_MEAN": float(df["ABS_ERROR"].mean()) if not df.empty else np.nan,
        "ABS_ERROR_MEDIAN": float(df["ABS_ERROR"].median()) if not df.empty else np.nan,
        "PREDICTION_STD": float(df["D+1"].std(ddof=1)) if len(df) > 1 else np.nan,
        "ACTUAL_STD": float(df["ACTUAL_VALUE"].std(ddof=1)) if len(df) > 1 else np.nan,
        "ERROR_AUTOCORR_LAG1": float(error.autocorr(lag=1)) if len(error) > 2 else np.nan,
    }


def compute_spike_metrics(
    df: pd.DataFrame,
    model_row: pd.Series,
    spike_quantile: float,
) -> dict:
    """
    Compute performance on spike periods.

    Parameters:
        df: Processed result DataFrame.
        model_row: Summary row.
        spike_quantile: Quantile threshold for spikes.

    Returns:
        Dictionary of spike metrics.
    """
    threshold = float(df["ACTUAL_VALUE"].quantile(spike_quantile))
    spike_df = df.loc[df["ACTUAL_VALUE"] >= threshold].copy()
    non_spike_df = df.loc[df["ACTUAL_VALUE"] < threshold].copy()

    def rmse(x: pd.DataFrame) -> float:
        return float(np.sqrt(x["SQUARED_ERROR"].mean())) if not x.empty else np.nan

    def mae(x: pd.DataFrame) -> float:
        return float(x["ABS_ERROR"].mean()) if not x.empty else np.nan

    overall_rmse = rmse(df)
    spike_rmse = rmse(spike_df)

    return {
        "MODEL": model_row["MODEL"],
        "MODEL_BASE": model_row["MODEL_BASE"],
        "MODEL_NAME": model_row["MODEL_NAME"],
        "MODEL_GROUP": model_row["MODEL_GROUP"],
        "PRODUCT": model_row["PRODUCT"],
        "PRODUCT_SIGN": model_row["PRODUCT_SIGN"],
        "PRICE_TYPE": model_row["PRICE_TYPE"],
        "LOOKBACK": model_row["LOOKBACK"],
        "STRATEGY": model_row["STRATEGY"],
        "SPIKE_QUANTILE": spike_quantile,
        "SPIKE_THRESHOLD": threshold,
        "N_ALL": int(len(df)),
        "N_SPIKES": int(len(spike_df)),
        "N_NON_SPIKES": int(len(non_spike_df)),
        "RMSE_ALL": overall_rmse,
        "MAE_ALL": mae(df),
        "RMSE_SPIKES": spike_rmse,
        "MAE_SPIKES": mae(spike_df),
        "RMSE_NON_SPIKES": rmse(non_spike_df),
        "MAE_NON_SPIKES": mae(non_spike_df),
        "SPIKE_RMSE_PREMIUM": (
            float(spike_rmse / overall_rmse)
            if pd.notna(spike_rmse) and pd.notna(overall_rmse) and overall_rmse != 0
            else np.nan
        ),
    }


def compute_level_bin_metrics(
    df: pd.DataFrame,
    model_row: pd.Series,
    n_level_bins: int,
) -> pd.DataFrame:
    """
    Compute error by ACTUAL_VALUE bins.

    Parameters:
        df: Processed result DataFrame.
        model_row: Summary row.
        n_level_bins: Number of bins.

    Returns:
        Level-bin metrics DataFrame.
    """
    work = df.copy()

    if work["ACTUAL_VALUE"].nunique(dropna=True) < n_level_bins:
        work["LEVEL_BIN"] = "all"
    else:
        work["LEVEL_BIN"] = pd.qcut(
            work["ACTUAL_VALUE"],
            q=n_level_bins,
            duplicates="drop",
        ).astype(str)

    rows: list[dict] = []

    for level_bin, subset in work.groupby("LEVEL_BIN", dropna=False):
        rows.append(
            {
                "MODEL": model_row["MODEL"],
                "MODEL_BASE": model_row["MODEL_BASE"],
                "MODEL_NAME": model_row["MODEL_NAME"],
                "MODEL_GROUP": model_row["MODEL_GROUP"],
                "PRODUCT": model_row["PRODUCT"],
                "PRODUCT_SIGN": model_row["PRODUCT_SIGN"],
                "PRICE_TYPE": model_row["PRICE_TYPE"],
                "LOOKBACK": model_row["LOOKBACK"],
                "STRATEGY": model_row["STRATEGY"],
                "LEVEL_BIN": level_bin,
                "N_OBSERVATIONS": int(len(subset)),
                "BIN_MIN_ACTUAL": float(subset["ACTUAL_VALUE"].min()) if not subset.empty else np.nan,
                "BIN_MAX_ACTUAL": float(subset["ACTUAL_VALUE"].max()) if not subset.empty else np.nan,
                "BIN_MEAN_ACTUAL": float(subset["ACTUAL_VALUE"].mean()) if not subset.empty else np.nan,
                "MAE": float(subset["ABS_ERROR"].mean()) if not subset.empty else np.nan,
                "RMSE": float(np.sqrt(subset["SQUARED_ERROR"].mean())) if not subset.empty else np.nan,
                "MEAN_ERROR": float(subset["ERROR"].mean()) if not subset.empty else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_directional_metrics(
    df: pd.DataFrame,
    model_row: pd.Series,
) -> dict:
    """
    Compute directional accuracy metrics.

    Parameters:
        df: Processed result DataFrame.
        model_row: Summary row.

    Returns:
        Dictionary of directional metrics.
    """
    work = df.copy()

    valid = work.loc[
        work["PRED_CHANGE"].notna() &
        work["ACTUAL_CHANGE"].notna()
    ].copy()

    if valid.empty:
        directional_accuracy = np.nan
        hit_count = 0
        total_count = 0
    else:
        valid["PRED_DIRECTION"] = np.sign(valid["PRED_CHANGE"])
        valid["ACTUAL_DIRECTION"] = np.sign(valid["ACTUAL_CHANGE"])
        valid["DIRECTION_HIT"] = (valid["PRED_DIRECTION"] == valid["ACTUAL_DIRECTION"]).astype(int)

        hit_count = int(valid["DIRECTION_HIT"].sum())
        total_count = int(len(valid))
        directional_accuracy = float(valid["DIRECTION_HIT"].mean())

    return {
        "MODEL": model_row["MODEL"],
        "MODEL_BASE": model_row["MODEL_BASE"],
        "MODEL_NAME": model_row["MODEL_NAME"],
        "MODEL_GROUP": model_row["MODEL_GROUP"],
        "PRODUCT": model_row["PRODUCT"],
        "PRODUCT_SIGN": model_row["PRODUCT_SIGN"],
        "PRICE_TYPE": model_row["PRICE_TYPE"],
        "LOOKBACK": model_row["LOOKBACK"],
        "STRATEGY": model_row["STRATEGY"],
        "N_DIRECTIONAL_OBSERVATIONS": total_count,
        "DIRECTIONAL_HIT_COUNT": hit_count,
        "DIRECTIONAL_ACCURACY": directional_accuracy,
    }


def compute_ensemble_effects(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantify the contribution of LSTM/BiLSTM ensemble strategies.

    Parameters:
        master_df: Master comparison DataFrame.

    Returns:
        Ensemble effects DataFrame.
    """
    deep_models = master_df.loc[
        master_df["MODEL_BASE"].isin(
            [
                "lstm",
                "bidirectional_lstm",
                "lstm_features_1",
                "lstm_features_4",
                "lstm_features_1_3",
                "lstm_features_1_4",
            ]
        )
    ].copy()

    rows: list[dict] = []
    group_cols = ["MODEL_BASE", "PRODUCT", "PRODUCT_SIGN", "PRICE_TYPE", "LOOKBACK"]

    for _, subset in deep_models.groupby(group_cols, dropna=False):
        strategy_map: dict[str, pd.Series] = {}

        for _, row in subset.iterrows():
            strategy = str(row["STRATEGY"])
            strategy_map[strategy] = row

        if "single" not in strategy_map:
            continue

        single_row = strategy_map["single"]

        for candidate_strategy in ["ensemble", "recency"]:
            if candidate_strategy not in strategy_map:
                continue

            candidate_row = strategy_map[candidate_strategy]

            rows.append(
                {
                    "MODEL_BASE": single_row["MODEL_BASE"],
                    "PRODUCT": single_row["PRODUCT"],
                    "PRODUCT_SIGN": single_row["PRODUCT_SIGN"],
                    "PRICE_TYPE": single_row["PRICE_TYPE"],
                    "LOOKBACK": single_row["LOOKBACK"],
                    "BASELINE_STRATEGY": "single",
                    "COMPARE_STRATEGY": candidate_strategy,
                    "RMSE_SINGLE": float(single_row["RMSE"]),
                    "RMSE_COMPARE": float(candidate_row["RMSE"]),
                    "MAE_SINGLE": float(single_row["MAE"]),
                    "MAE_COMPARE": float(candidate_row["MAE"]),
                    "REVENUE_SINGLE": float(single_row["REVENUE"]),
                    "REVENUE_COMPARE": float(candidate_row["REVENUE"]),
                    "RMSE_IMPROVEMENT_PCT": (
                        ((float(single_row["RMSE"]) - float(candidate_row["RMSE"])) / float(single_row["RMSE"])) * 100.0
                        if pd.notna(single_row["RMSE"]) and pd.notna(candidate_row["RMSE"]) and float(single_row["RMSE"]) != 0
                        else np.nan
                    ),
                    "MAE_IMPROVEMENT_PCT": (
                        ((float(single_row["MAE"]) - float(candidate_row["MAE"])) / float(single_row["MAE"])) * 100.0
                        if pd.notna(single_row["MAE"]) and pd.notna(candidate_row["MAE"]) and float(single_row["MAE"]) != 0
                        else np.nan
                    ),
                    "REVENUE_IMPROVEMENT_PCT": (
                        ((float(candidate_row["REVENUE"]) - float(single_row["REVENUE"])) / abs(float(single_row["REVENUE"]))) * 100.0
                        if pd.notna(single_row["REVENUE"]) and pd.notna(candidate_row["REVENUE"]) and float(single_row["REVENUE"]) != 0
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def compute_reference_comparison(
    master_df: pd.DataFrame,
    reference_models: list[str],
) -> pd.DataFrame:
    """
    Compare a selected benchmark set across identical combinations.

    Parameters:
        master_df: Master comparison DataFrame.
        reference_models: Models to compare.

    Returns:
        Filtered comparison table.
    """
    subset = master_df.loc[master_df["MODEL_BASE"].isin(reference_models)].copy()

    desired_order = {model: i for i, model in enumerate(reference_models)}
    subset["MODEL_ORDER"] = subset["MODEL_BASE"].map(desired_order)
    subset = subset.sort_values(
        by=["PRODUCT", "PRODUCT_SIGN", "PRICE_TYPE", "MODEL_ORDER", "LOOKBACK", "STRATEGY"]
    ).drop(columns=["MODEL_ORDER"])

    return subset


def print_focus_fairness_check(
    master_df: pd.DataFrame,
    focus_product: str,
    focus_product_sign: str,
    focus_price_type: str,
) -> None:
    """
    Print a compact fairness check for one exact combination.

    Parameters:
        master_df: Master comparison DataFrame.
        focus_product: Product block.
        focus_product_sign: Product sign.
        focus_price_type: Price type.

    Returns:
        None
    """
    subset = master_df.loc[
        (master_df["PRODUCT"] == focus_product) &
        (master_df["PRODUCT_SIGN"] == focus_product_sign) &
        (master_df["PRICE_TYPE"] == focus_price_type)
    ].copy()

    if subset.empty:
        print("\nNo rows found for focus fairness check.")
        return

    subset = subset[
        [
            "MODEL",
            "MODEL_BASE",
            "MODEL_GROUP",
            "LOOKBACK",
            "STRATEGY",
            "N_OBSERVATIONS",
        ]
    ].sort_values(by=["MODEL_GROUP", "MODEL"])

    unique_counts = sorted(subset["N_OBSERVATIONS"].dropna().unique().tolist())

    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK: N_OBSERVATIONS")
    print("=" * 80)
    print(f"\nSubset used for check ({focus_product} / {focus_product_sign} / {focus_price_type}):\n")
    print(subset.to_string(index=False))

    print("\n--- RESULT ---")
    if len(unique_counts) == 1:
        print(f"✅ All models use the SAME number of observations: {unique_counts[0]}")
    else:
        print(f"⚠️ Different observation counts found across models: {unique_counts}")


def print_focus_date_check(
    postprocessed_root: Path,
    master_df: pd.DataFrame,
    focus_product: str,
    focus_product_sign: str,
    focus_price_type: str,
) -> None:
    """
    Print a compact DATE-range consistency check for one exact combination.

    Parameters:
        postprocessed_root: Postprocessed root folder.
        master_df: Master comparison DataFrame.
        focus_product: Product block.
        focus_product_sign: Product sign.
        focus_price_type: Price type.

    Returns:
        None
    """
    subset = master_df.loc[
        (master_df["PRODUCT"] == focus_product) &
        (master_df["PRODUCT_SIGN"] == focus_product_sign) &
        (master_df["PRICE_TYPE"] == focus_price_type)
    ].copy()

    if subset.empty:
        print("\nNo rows found for focus DATE check.")
        return

    rows: list[dict] = []

    for _, row in subset.iterrows():
        file_path = build_processed_result_path(
            postprocessed_root=postprocessed_root,
            model_base=str(row["MODEL_BASE"]),
            product=str(row["PRODUCT"]),
            product_sign=str(row["PRODUCT_SIGN"]),
            price_type=str(row["PRICE_TYPE"]),
            lookback=(int(row["LOOKBACK"]) if pd.notna(row["LOOKBACK"]) else None),
            strategy=(str(row["STRATEGY"]) if pd.notna(row["STRATEGY"]) else None),
        )

        if not file_path.exists():
            rows.append(
                {
                    "MODEL": row["MODEL"],
                    "DATE_MIN": pd.NaT,
                    "DATE_MAX": pd.NaT,
                    "N_DATES": np.nan,
                    "FILE_EXISTS": False,
                }
            )
            continue

        df = load_processed_result_file(file_path)
        rows.append(
            {
                "MODEL": row["MODEL"],
                "DATE_MIN": df.index.min(),
                "DATE_MAX": df.index.max(),
                "N_DATES": int(len(df)),
                "FILE_EXISTS": True,
            }
        )

    check_df = pd.DataFrame(rows).sort_values(by="MODEL")

    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK: DATE RANGE")
    print("=" * 80)
    print(f"\nSubset used for check ({focus_product} / {focus_product_sign} / {focus_price_type}):\n")
    print(check_df.to_string(index=False))

    existing = check_df.loc[check_df["FILE_EXISTS"]].copy()

    print("\n--- RESULT ---")
    if existing.empty:
        print("⚠️ No processed files found for DATE check.")
        return

    unique_mins = existing["DATE_MIN"].dropna().unique()
    unique_maxs = existing["DATE_MAX"].dropna().unique()
    unique_n = existing["N_DATES"].dropna().unique()

    if len(unique_mins) == 1 and len(unique_maxs) == 1 and len(unique_n) == 1:
        print(
            f"✅ All models share identical DATE range: "
            f"{existing['DATE_MIN'].iloc[0]} → {existing['DATE_MAX'].iloc[0]}"
        )
    else:
        print("⚠️ DATE inconsistency detected across models.")
        print(f"Unique DATE_MIN values: {list(unique_mins)}")
        print(f"Unique DATE_MAX values: {list(unique_maxs)}")
        print(f"Unique N_DATES values: {list(unique_n)}")


def interpret_and_decompose(
    bias_df: pd.DataFrame,
    spike_df: pd.DataFrame,
    master_df: pd.DataFrame,
) -> None:
    """
    Print quantified interpretation of why one model group may outperform others.

    Parameters:
        bias_df: Bias/variance diagnostics table.
        spike_df: Spike diagnostics table.
        master_df: Master comparison table.

    Returns:
        None
    """
    print("\n" + "=" * 80)
    print("INTERPRETATION + DECOMPOSITION")
    print("=" * 80)

    if bias_df.empty or spike_df.empty or master_df.empty:
        print("⚠️ Missing diagnostic data, cannot interpret results.")
        return

    bias_summary = bias_df.groupby("MODEL_GROUP", dropna=False).agg(
        {
            "ABS_ERROR_MEAN": "mean",
            "ERROR_STD": "mean",
            "MEAN_ERROR_BIAS": "mean",
        }
    )

    spike_summary = spike_df.groupby("MODEL_GROUP", dropna=False).agg(
        {
            "RMSE_SPIKES": "mean",
            "MAE_SPIKES": "mean",
            "SPIKE_RMSE_PREMIUM": "mean",
        }
    )

    rmse_summary = master_df.groupby("MODEL_GROUP", dropna=False).agg(
        {
            "RMSE": "mean",
            "MAE": "mean",
            "REVENUE": "mean",
        }
    )

    summary = bias_summary.join(spike_summary, how="outer").join(rmse_summary, how="outer")
    summary = summary.sort_index()

    print("\nAggregated model-group diagnostics:\n")
    print(summary.round(6).to_string())

    statistical_groups = ["STATISTICAL_EXPANDING", "STATISTICAL_ROLLING"]
    statistical_ref = summary.loc[summary.index.intersection(statistical_groups)].copy()

    if statistical_ref.empty:
        print("\n⚠️ No statistical reference group found. Decomposition skipped.")
        return

    ref_rmse = float(statistical_ref["RMSE"].min())
    ref_error_std = float(statistical_ref["ERROR_STD"].mean())
    ref_spike_rmse = float(statistical_ref["RMSE_SPIKES"].mean())
    ref_abs_bias = float(statistical_ref["MEAN_ERROR_BIAS"].abs().mean())

    print("\nReference values built from statistical models:")
    print(f"Best statistical mean RMSE: {ref_rmse:.6f}")
    print(f"Average statistical error std: {ref_error_std:.6f}")
    print(f"Average statistical spike RMSE: {ref_spike_rmse:.6f}")
    print(f"Average absolute statistical bias: {ref_abs_bias:.6f}")

    print("\n--- GROUP-BY-GROUP DECOMPOSITION ---")

    for group_name, row in summary.iterrows():
        if pd.isna(row.get("RMSE")):
            continue

        total_rmse_gain = ref_rmse - float(row["RMSE"])

        print(f"\nModel group: {group_name}")
        print(f"Mean RMSE: {float(row['RMSE']):.6f}")
        print(f"Mean MAE: {float(row['MAE']):.6f}" if pd.notna(row.get("MAE")) else "Mean MAE: nan")
        print(f"Mean Revenue: {float(row['REVENUE']):.6f}" if pd.notna(row.get("REVENUE")) else "Mean Revenue: nan")
        print(f"RMSE gain vs best statistical reference: {total_rmse_gain:.6f}")

        variance_gain = ref_error_std - float(row["ERROR_STD"]) if pd.notna(row.get("ERROR_STD")) else np.nan
        spike_gain = ref_spike_rmse - float(row["RMSE_SPIKES"]) if pd.notna(row.get("RMSE_SPIKES")) else np.nan
        bias_gain = ref_abs_bias - abs(float(row["MEAN_ERROR_BIAS"])) if pd.notna(row.get("MEAN_ERROR_BIAS")) else np.nan

        components = {
            "VARIANCE_REDUCTION": variance_gain,
            "SPIKE_HANDLING": spike_gain,
            "BIAS_REDUCTION": bias_gain,
        }

        valid_components = {k: v for k, v in components.items() if pd.notna(v)}
        denom = sum(abs(v) for v in valid_components.values())

        if denom == 0:
            print("No decomposition possible because all component gains are zero.")
            continue

        for component_name, value in valid_components.items():
            pct = (value / denom) * 100.0
            print(f"{component_name}: {value:.6f} ({pct:.2f}%)")

        dominant_component = max(valid_components.items(), key=lambda item: abs(item[1]))[0]
        print(f"👉 Dominant quantitative driver: {dominant_component}")

def print_csv_conclusions(
    diagnostics_root: Path,
    bias_variance_df: pd.DataFrame,
    spike_df: pd.DataFrame,
    level_bin_df: pd.DataFrame,
    directional_df: pd.DataFrame,
    ensemble_effects_df: pd.DataFrame,
    reference_comparison_df: pd.DataFrame,
) -> None:
    """
    Print strong quantitative conclusions for each diagnostic CSV.
    """

    print("\n" + "=" * 80)
    print("QUANTITATIVE MODEL INSIGHTS")
    print("=" * 80)

    # ----------------------------------------------------------------------------------
    # 1) BIAS / VARIANCE
    # ----------------------------------------------------------------------------------
    print(f"\n1) bias_variance_diagnostics.csv")

    if not bias_variance_df.empty:

        summary = bias_variance_df.groupby("MODEL_GROUP").agg({
            "ABS_ERROR_MEAN": "mean",
            "ERROR_STD": "mean",
            "MEAN_ERROR_BIAS": "mean",
        })

        summary = summary.sort_values("ABS_ERROR_MEAN")

        print("\nModel group comparison:")
        print(summary.round(6).to_string())

        best = summary.iloc[0]
        worst = summary.iloc[-1]

        improvement = (worst["ABS_ERROR_MEAN"] - best["ABS_ERROR_MEAN"]) / worst["ABS_ERROR_MEAN"] * 100

        print("\n👉 Insight:")
        print(
            f"{summary.index[0]} has the lowest error.\n"
            f"It improves MAE by {improvement:.2f}% vs worst group ({summary.index[-1]})."
        )

        print(
            f"Variance (ERROR_STD): {summary.index[0]} = {best['ERROR_STD']:.4f} vs "
            f"{summary.index[-1]} = {worst['ERROR_STD']:.4f}"
        )

        print(
            f"Bias (MEAN_ERROR_BIAS): {summary.index[0]} = {best['MEAN_ERROR_BIAS']:.4f}"
        )

    # ----------------------------------------------------------------------------------
    # 2) SPIKES
    # ----------------------------------------------------------------------------------
    print(f"\n2) spike_diagnostics.csv")

    if not spike_df.empty:

        summary = spike_df.groupby("MODEL_GROUP").agg({
            "RMSE_SPIKES": "mean",
            "SPIKE_RMSE_PREMIUM": "mean",
        })

        summary = summary.sort_values("RMSE_SPIKES")

        print("\nSpike performance:")
        print(summary.round(6).to_string())

        best = summary.iloc[0]
        worst = summary.iloc[-1]

        improvement = (worst["RMSE_SPIKES"] - best["RMSE_SPIKES"]) / worst["RMSE_SPIKES"] * 100

        print("\n👉 Insight:")
        print(
            f"{summary.index[0]} handles spikes best.\n"
            f"Spike RMSE improves by {improvement:.2f}% vs worst group ({summary.index[-1]})."
        )

        print(
            f"Spike penalty (premium): {summary.index[0]} = {best['SPIKE_RMSE_PREMIUM']:.2f}"
        )

    # ----------------------------------------------------------------------------------
    # 3) LEVEL BINS
    # ----------------------------------------------------------------------------------
    print(f"\n3) level_bin_diagnostics.csv")

    if not level_bin_df.empty:

        high_bin = level_bin_df.copy()

        # focus on highest bin only
        high_bin = high_bin.sort_values("BIN_MEAN_ACTUAL").groupby("MODEL_GROUP").tail(1)

        high_bin_summary = high_bin.groupby("MODEL_GROUP")[["MAE", "RMSE"]].mean()
        high_bin_summary = high_bin_summary.sort_values("MAE")

        print("\nHigh-price regime comparison:")
        print(high_bin_summary.round(6).to_string())

        best = high_bin_summary.iloc[0]
        worst = high_bin_summary.iloc[-1]

        improvement = (worst["MAE"] - best["MAE"]) / worst["MAE"] * 100

        print("\n👉 Insight:")
        print(
            f"{high_bin_summary.index[0]} performs best in extreme price regimes.\n"
            f"MAE improves by {improvement:.2f}% vs worst group."
        )

    # ----------------------------------------------------------------------------------
    # 4) DIRECTION
    # ----------------------------------------------------------------------------------
    print(f"\n4) directional_accuracy_diagnostics.csv")

    if not directional_df.empty:

        summary = directional_df.groupby("MODEL_GROUP")["DIRECTIONAL_ACCURACY"].mean()
        summary = summary.sort_values(ascending=False)

        print("\nDirectional accuracy:")
        print(summary.round(4).to_string())

        best = summary.iloc[0]
        worst = summary.iloc[-1]

        diff = (best - worst) * 100

        print("\n👉 Insight:")
        print(
            f"{summary.index[0]} predicts direction best.\n"
            f"Accuracy advantage: {diff:.2f} percentage points vs worst group."
        )

    # ----------------------------------------------------------------------------------
    # 5) ENSEMBLE EFFECT
    # ----------------------------------------------------------------------------------
    print(f"\n5) ensemble_effect_diagnostics.csv")

    if not ensemble_effects_df.empty:

        summary = ensemble_effects_df.groupby("COMPARE_STRATEGY").agg({
            "RMSE_IMPROVEMENT_PCT": "mean",
            "MAE_IMPROVEMENT_PCT": "mean",
        })

        print("\nEnsemble vs single:")
        print(summary.round(4).to_string())

        best_strategy = summary["RMSE_IMPROVEMENT_PCT"].idxmax()

        print("\n👉 Insight:")
        print(
            f"{best_strategy} improves performance the most over single models.\n"
            f"Average RMSE improvement: {summary.loc[best_strategy, 'RMSE_IMPROVEMENT_PCT']:.2f}%"
        )

    # ----------------------------------------------------------------------------------
    # 6) DIRECT MODEL COMPARISON (VERY IMPORTANT)
    # ----------------------------------------------------------------------------------
    print(f"\n6) benchmark_reference_comparison.csv")

    if not reference_comparison_df.empty:

        summary = reference_comparison_df.groupby("MODEL_BASE").agg({
            "RMSE": "mean",
            "MAE": "mean",
            "REVENUE": "mean",
        })

        summary = summary.sort_values("RMSE")

        print("\nBenchmark comparison:")
        print(summary.round(6).to_string())

        best = summary.iloc[0]
        worst = summary.iloc[-1]

        improvement = (worst["RMSE"] - best["RMSE"]) / worst["RMSE"] * 100

        print("\n👉 Insight:")
        print(
            f"{summary.index[0]} is best overall among benchmark models.\n"
            f"RMSE improvement: {improvement:.2f}% vs {summary.index[-1]}"
        )

        print(
            f"Revenue comparison: best model = {summary['REVENUE'].idxmax()}"
        )


# --------------------------------------------------------------------------------------
# Main execution
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

    config = DiagnosticConfig(
        postprocessed_root=BASE_OUTPUT_PATH / "postprocessed_data",
        diagnostics_root=BASE_OUTPUT_PATH / "postprocessed_data" / "diagnostics",
    )
    config.diagnostics_root.mkdir(parents=True, exist_ok=True)

    master_df = load_master_comparison(config.postprocessed_root)

    if master_df.empty:
        raise ValueError("all_models_comparison.csv is empty. Run results_postprocessing first.")

    bias_variance_rows: list[dict] = []
    spike_rows: list[dict] = []
    level_bin_tables: list[pd.DataFrame] = []
    directional_rows: list[dict] = []

    LOGGER.info("Starting per-model diagnostics using processed result files.")

    for _, model_row in master_df.iterrows():
        lookback_value: Optional[int] = (
            int(model_row["LOOKBACK"])
            if "LOOKBACK" in model_row and pd.notna(model_row["LOOKBACK"])
            else None
        )
        strategy_value: Optional[str] = (
            str(model_row["STRATEGY"])
            if "STRATEGY" in model_row and pd.notna(model_row["STRATEGY"])
            else None
        )

        processed_path = build_processed_result_path(
            postprocessed_root=config.postprocessed_root,
            model_base=str(model_row["MODEL_BASE"]),
            product=str(model_row["PRODUCT"]),
            product_sign=str(model_row["PRODUCT_SIGN"]),
            price_type=str(model_row["PRICE_TYPE"]),
            lookback=lookback_value,
            strategy=strategy_value,
        )

        if not processed_path.exists():
            LOGGER.warning("Processed file missing, skipping diagnostics: %s", processed_path)
            continue

        processed_df = load_processed_result_file(processed_path)
        processed_df = processed_df.loc[
            processed_df.index >= pd.Timestamp(config.min_date)
        ].copy()

        if processed_df.empty:
            LOGGER.warning("Processed file empty after date filter: %s", processed_path)
            continue

        processed_df = add_error_columns(processed_df)

        bias_variance_rows.append(
            compute_bias_variance_metrics(
                df=processed_df,
                model_row=model_row,
            )
        )
        spike_rows.append(
            compute_spike_metrics(
                df=processed_df,
                model_row=model_row,
                spike_quantile=config.spike_quantile,
            )
        )
        level_bin_tables.append(
            compute_level_bin_metrics(
                df=processed_df,
                model_row=model_row,
                n_level_bins=config.n_level_bins,
            )
        )
        directional_rows.append(
            compute_directional_metrics(
                df=processed_df,
                model_row=model_row,
            )
        )

    bias_variance_df = pd.DataFrame(bias_variance_rows)
    spike_df = pd.DataFrame(spike_rows)
    level_bin_df = pd.concat(level_bin_tables, ignore_index=True) if level_bin_tables else pd.DataFrame()
    directional_df = pd.DataFrame(directional_rows)
    ensemble_effects_df = compute_ensemble_effects(master_df=master_df)

    benchmark_models = [
        "arma",
        "armax_exog_4",
        "lstm",
        "xgboost_exog_4",
        "xgboost_v2_exog_4",
    ]
    reference_comparison_df = compute_reference_comparison(
        master_df=master_df,
        reference_models=benchmark_models,
    )

    if not bias_variance_df.empty:
        bias_variance_df.to_csv(config.diagnostics_root / "bias_variance_diagnostics.csv", index=False)
    if not spike_df.empty:
        spike_df.to_csv(config.diagnostics_root / "spike_diagnostics.csv", index=False)
    if not level_bin_df.empty:
        level_bin_df.to_csv(config.diagnostics_root / "level_bin_diagnostics.csv", index=False)
    if not directional_df.empty:
        directional_df.to_csv(config.diagnostics_root / "directional_accuracy_diagnostics.csv", index=False)
    if not ensemble_effects_df.empty:
        ensemble_effects_df.to_csv(config.diagnostics_root / "ensemble_effect_diagnostics.csv", index=False)
    if not reference_comparison_df.empty:
        reference_comparison_df.to_csv(config.diagnostics_root / "benchmark_reference_comparison.csv", index=False)

    LOGGER.info("Saved diagnostic tables to: %s", config.diagnostics_root)

    interpret_and_decompose(
        bias_df=bias_variance_df,
        spike_df=spike_df,
        master_df=master_df,
    )

    print_csv_conclusions(
        diagnostics_root=config.diagnostics_root,
        bias_variance_df=bias_variance_df,
        spike_df=spike_df,
        level_bin_df=level_bin_df,
        directional_df=directional_df,
        ensemble_effects_df=ensemble_effects_df,
        reference_comparison_df=reference_comparison_df,
    )

    print_focus_fairness_check(
        master_df=master_df,
        focus_product=config.focus_product,
        focus_product_sign=config.focus_product_sign,
        focus_price_type=config.focus_price_type,
    )

    print_focus_date_check(
        postprocessed_root=config.postprocessed_root,
        master_df=master_df,
        focus_product=config.focus_product,
        focus_product_sign=config.focus_product_sign,
        focus_price_type=config.focus_price_type,
    )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC OUTPUTS CREATED")
    print("=" * 80)
    print(f"bias_variance_diagnostics.csv        -> {config.diagnostics_root / 'bias_variance_diagnostics.csv'}")
    print(f"spike_diagnostics.csv                -> {config.diagnostics_root / 'spike_diagnostics.csv'}")
    print(f"level_bin_diagnostics.csv            -> {config.diagnostics_root / 'level_bin_diagnostics.csv'}")
    print(f"directional_accuracy_diagnostics.csv -> {config.diagnostics_root / 'directional_accuracy_diagnostics.csv'}")
    print(f"ensemble_effect_diagnostics.csv      -> {config.diagnostics_root / 'ensemble_effect_diagnostics.csv'}")
    print(f"benchmark_reference_comparison.csv   -> {config.diagnostics_root / 'benchmark_reference_comparison.csv'}")
    print("\nDone. Use these tables and printed conclusions to quantify why LSTM wins.")


# Example usage
# python3 -m postprocessing.model_diagnostics
if __name__ == "__main__":
    main()