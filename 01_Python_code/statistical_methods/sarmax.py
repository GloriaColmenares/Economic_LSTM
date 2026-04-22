# 01_Python_code/statistical_methods/sarmax.py

from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------------------
# Limit threading (VERY IMPORTANT for stability)
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# --------------------------------------------------------------------------------------
# Import paths
# --------------------------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
PROJECT_CODE_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE_ROOT))

from paths import BASE_INPUT_PATH, BASE_OUTPUT_PATH  # noqa

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


@dataclass
class SarmaxResult:
    product: str
    product_sign: str
    price_type: str
    feature_nums: str
    order: str
    seasonal_order: str
    aic: Optional[float]
    output_file: str
    success: bool
    message: str


# --------------------------------------------------------------------------------------
# Core function
# --------------------------------------------------------------------------------------

def sarmax_forecast(
    dataset: pd.DataFrame,
    exo_dataset: pd.DataFrame,
    product: str,
    product_sign: str,
    price_type: str,
    feature_nums: list[int],
    output_folder: Path,
) -> SarmaxResult:

    feature_nums_str = "_".join(map(str, feature_nums))
    target_col = f"{product_sign}_GERMANY_{price_type}_CAPACITY_PRICE_[(EUR/MW)/h]"
    result_file = output_folder / f"sarmax_exog_{feature_nums_str}_model_results_{product}_{product_sign}_{price_type}.csv"

    try:
        subset = dataset.loc[dataset["PRODUCT"] == product].copy()
        exo_subset = exo_dataset.loc[exo_dataset["PRODUCT"] == product].drop(columns=["PRODUCT"]).copy()

        if subset.empty or exo_subset.empty:
            raise ValueError("Empty dataset after filtering")

        df = subset[[target_col]].dropna().copy()

        # Align indices
        common_idx = df.index.intersection(exo_subset.index)
        df = df.loc[common_idx]
        exo_subset = exo_subset.loc[common_idx]

        if len(df) < 20:
            raise ValueError("Not enough data")

        # Split
        n = len(df)
        train_end = int(n * 0.7)
        valid_end = int(n * 0.85)

        train = df.iloc[:train_end]
        valid = df.iloc[train_end:valid_end]
        test = df.iloc[valid_end:]

        exo_train = exo_subset.iloc[:train_end]
        exo_valid = exo_subset.iloc[train_end:valid_end]
        exo_test = exo_subset.iloc[valid_end:]

        # -------------------------
        # Preprocessing (TRAIN ONLY)
        # -------------------------
        cap = train.quantile(0.95).iloc[0]
        train = train.clip(upper=cap)
        valid = valid.clip(upper=cap)
        test = test.clip(upper=cap)

        if (train.iloc[:, 0] <= 0).any():
            raise ValueError("Negative values for log")

        train = np.log(train)
        valid = np.log(valid)
        test = np.log(test)

        # -------------------------
        # Model selection
        # -------------------------
        auto_model = auto_arima(
            train.iloc[:, 0],
            seasonal=True,
            m=7,
            d=0,
            D=0,
            stepwise=False,
            n_jobs=1,
            suppress_warnings=True,
            error_action="ignore",
        )

        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        aic = auto_model.aic()

        # -------------------------
        # Forecast (expanding)
        # -------------------------
        def rolling_forecast(y_hist, x_hist, y_future, x_future):
            y_hist = list(y_hist.values)
            x_hist = list(x_hist.values)

            preds = []

            for i in range(len(y_future)):
                model = SARIMAX(
                    endog=np.array(y_hist),
                    exog=np.array(x_hist),
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                )
                fit = model.fit(disp=False)

                pred = fit.forecast(
                    steps=1,
                    exog=x_future.iloc[i:i+1].values
                )[0]

                preds.append(pred)

                y_hist.append(y_future.iloc[i])
                x_hist.append(x_future.iloc[i].values)

            return preds

        valid_pred = rolling_forecast(train.iloc[:, 0], exo_train, valid.iloc[:, 0], exo_valid)
        test_pred = rolling_forecast(
            pd.concat([train, valid]).iloc[:, 0],
            pd.concat([exo_train, exo_valid]),
            test.iloc[:, 0],
            exo_test
        )

        # Inverse log
        valid_pred = np.exp(valid_pred)
        test_pred = np.exp(test_pred)

        # Output
        valid_df = pd.DataFrame({
            "DATE": valid.index,
            "ACTUAL_VALUE": df.iloc[train_end:valid_end, 0].values,
            "D+1": valid_pred,
        })

        test_df = pd.DataFrame({
            "DATE": test.index,
            "ACTUAL_VALUE": df.iloc[valid_end:, 0].values,
            "D+1": test_pred,
        })

        pd.concat([valid_df, test_df]).to_csv(result_file, index=False)

        return SarmaxResult(
            product, product_sign, price_type,
            feature_nums_str,
            str(order),
            str(seasonal_order),
            aic,
            str(result_file),
            True,
            "success"
        )

    except Exception as e:
        LOGGER.exception("Error")
        return SarmaxResult(
            product, product_sign, price_type,
            feature_nums_str,
            "", "", None,
            str(result_file),
            False,
            str(e)
        )


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

def main() -> None:
    configure_logging()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    products = ['00_04','04_08','08_12','12_16','16_20','20_24']
    product_signs = ['NEG','POS']
    price_types = ['AVERAGE','MARGINAL']
    
    #feature_nums = [4]
    #feature_nums = [1]
    #feature_nums = [1,4]
    feature_nums = [1,3]

    exog_factors = {
        1: 'FCR_GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]',
        2: 'CUMULATED_CAPACITY_MORE_EXPENSIVE_THAN_ELECTRICITY_PRICE_EXCESS_CAPACITY',
        3: 'RES_SHARE',
        4: 'EXESSIVE_AVAILABLE_CAPACITY_UNTIL_PRICE_LIMIT_9999_€/MWh',
        5: 'CLEAN_SPREAD_OF_GAS'
    }

    selected_features = [exog_factors[i] for i in feature_nums]

    dataset = pd.read_csv(
        BASE_INPUT_PATH / "processed_afrr_data.csv",
        parse_dates=["DATE"],
        index_col="DATE"
    )

    exo = pd.read_csv(
        BASE_INPUT_PATH / "selected_exogenous_factors_20210101_20250228.csv",
        parse_dates=["DATE"],
        index_col="DATE"
    )

    exo = exo[["PRODUCT"] + selected_features]

    output_folder = BASE_OUTPUT_PATH / f"sarmax_exog_{'_'.join(map(str, feature_nums))}_model_output"
    output_folder.mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=2)(
        delayed(sarmax_forecast)(
            dataset, exo, p, s, pt, feature_nums, output_folder
        )
        for p in products
        for s in product_signs
        for pt in price_types
    )

    overview = pd.DataFrame([r.__dict__ for r in results])
    overview.to_csv(output_folder / f"sarmax_exog_{'_'.join(map(str, feature_nums))}_models_overview.csv", index=False)


if __name__ == "__main__":
    main()