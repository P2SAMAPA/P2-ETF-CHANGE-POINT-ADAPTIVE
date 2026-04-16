"""
Global and Adaptive Window training orchestration.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import config
from data_manager import load_master_data, prepare_data, get_universe_returns
from change_point_detector import universe_adaptive_start_date
from model import AdaptiveRidgeModel, prepare_training_data
from selector import select_top_etf
from push_results import push_daily_result


def evaluate_etf(ticker: str, returns: pd.DataFrame) -> dict:
    """Compute performance metrics for a given ETF ticker."""
    col = f"{ticker}_ret"
    if col not in returns.columns:
        return {}
    ret_series = returns[col].dropna()
    if len(ret_series) < 5:
        return {}

    ann_return = ret_series.mean() * config.TRADING_DAYS_PER_YEAR
    ann_vol = ret_series.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + ret_series).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    hit_rate = (ret_series > 0).mean()

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
    }


def train_global(universe: str, returns: pd.DataFrame) -> dict:
    """Global training with fixed 80/10/10 split."""
    print(f"\n--- Global Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    total_days = len(returns)
    train_end_idx = int(total_days * config.TRAIN_RATIO)
    val_end_idx = train_end_idx + int(total_days * config.VAL_RATIO)

    train_ret = returns.iloc[:train_end_idx]
    test_ret = returns.iloc[val_end_idx:]

    X, y, _ = prepare_training_data(returns, tickers,
                                    train_ret.index[0], train_ret.index[-1])
    if X is None:
        return {"ticker": None, "metrics": {}, "pred_return": None}

    model = AdaptiveRidgeModel().fit(X, y)
    predictions = model.predict_next_returns(returns, tickers, train_ret.index[-1])
    top_etf = select_top_etf(predictions)
    pred_return = predictions.get(top_etf) if top_etf else None

    metrics = evaluate_etf(top_etf, test_ret)
    print(f"  Selected ETF: {top_etf}, Predicted Return: {pred_return*100:.2f}%" if pred_return else f"  Selected ETF: {top_etf}")

    return {
        "ticker": top_etf,
        "pred_return": pred_return,
        "metrics": metrics,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
    }


def train_adaptive(universe: str, returns: pd.DataFrame) -> dict:
    """Adaptive window training based on change point detection."""
    print(f"\n--- Adaptive Training: {universe} ---")
    tickers = [col.replace("_ret", "") for col in returns.columns]
    if returns.empty:
        return {"ticker": None, "metrics": {}, "pred_return": None, "change_point_date": None, "lookback_days": 0}

    # Determine adaptive start date
    cp_date = universe_adaptive_start_date(returns)
    print(f"  Adaptive window starts: {cp_date.date()}")

    # Use data from cp_date up to 20 days before the end (to allow a test period)
    end_date = returns.index[-1] - pd.Timedelta(days=20)
    if end_date <= cp_date:
        end_date = returns.index[-1]

    train_mask = (returns.index >= cp_date) & (returns.index <= end_date)
    train_ret = returns.loc[train_mask]
    test_ret = returns.loc[returns.index > end_date]

    if len(train_ret) < config.MIN_TRAIN_DAYS:
        print(f"  Insufficient training days ({len(train_ret)}). Falling back to global.")
        return train_global(universe, returns)

    X, y, _ = prepare_training_data(returns, tickers, cp_date, end_date)
    if X is None:
        return {"ticker": None, "metrics": {}, "pred_return": None}

    model = AdaptiveRidgeModel().fit(X, y)
    predictions = model.predict_next_returns(returns, tickers, end_date)
    top_etf = select_top_etf(predictions)
    pred_return = predictions.get(top_etf) if top_etf else None

    metrics = evaluate_etf(top_etf, test_ret)
    lookback_days = (returns.index[-1] - cp_date).days

    print(f"  Selected ETF: {top_etf}, Predicted Return: {pred_return*100:.2f}%" if pred_return else f"  Selected ETF: {top_etf}")

    return {
        "ticker": top_etf,
        "pred_return": pred_return,
        "metrics": metrics,
        "change_point_date": cp_date.strftime("%Y-%m-%d"),
        "lookback_days": lookback_days,
        "test_start": test_ret.index[0].strftime("%Y-%m-%d"),
        "test_end": test_ret.index[-1].strftime("%Y-%m-%d"),
    }


def run_training():
    print("Loading data...")
    df_raw = load_master_data()
    df = prepare_data(df_raw)

    all_results = {}
    for universe in ["fi", "equity", "combined"]:
        print(f"\n{'='*50}\nProcessing {universe.upper()} universe\n{'='*50}")
        returns = get_universe_returns(df, universe)
        if returns.empty:
            continue
        global_res = train_global(universe, returns)
        adaptive_res = train_adaptive(universe, returns)
        all_results[universe] = {
            "global": global_res,
            "adaptive": adaptive_res,
        }
    return all_results


if __name__ == "__main__":
    output = run_training()
    if config.HF_TOKEN:
        push_daily_result(output)
    else:
        print("HF_TOKEN not set.")
