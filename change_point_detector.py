"""
Change point detection using ruptures (PELT algorithm).
"""
import numpy as np
import pandas as pd
import ruptures as rpt
import config


def detect_change_points_single(series: pd.Series) -> list:
    """
    Detect change points in a univariate series (e.g., ETF returns).
    Returns list of index positions (iloc) where changes occur.
    """
    if len(series) < config.MIN_TRAIN_DAYS:
        return []
    values = series.values.reshape(-1, 1)
    algo = rpt.Pelt(model=config.CP_MODEL, min_size=config.CP_MIN_DAYS_BETWEEN).fit(values)
    change_points = algo.predict(pen=config.CP_PENALTY)
    # The last point is always the end of the series; we exclude it
    return change_points[:-1]


def get_most_recent_change_point(series: pd.Series) -> pd.Timestamp:
    """
    Return the date of the most recent change point in the series.
    If none detected, return the start of the series.
    """
    cp_indices = detect_change_points_single(series)
    if not cp_indices:
        return series.index[0]
    last_cp_idx = cp_indices[-1]
    return series.index[last_cp_idx]


def universe_adaptive_start_date(returns: pd.DataFrame) -> pd.Timestamp:
    """
    Determine the adaptive training start date for a universe.
    Computes the most recent change point for each ETF, then takes the
    date where at least CP_CONSENSUS_FRACTION of ETFs agree a change occurred.
    Falls back to the earliest date in the data.
    """
    tickers = [col.replace("_ret", "") for col in returns.columns]
    change_dates = []
    for ticker in tickers:
        col = f"{ticker}_ret"
        if col in returns.columns:
            cp_date = get_most_recent_change_point(returns[col])
            change_dates.append(cp_date)

    if not change_dates:
        return returns.index[0]

    # Count frequency of each change date
    from collections import Counter
    date_counts = Counter(change_dates)
    threshold = int(len(tickers) * config.CP_CONSENSUS_FRACTION)

    # Find the most recent date that meets the consensus threshold
    sorted_dates = sorted(date_counts.keys(), reverse=True)
    for date in sorted_dates:
        if date_counts[date] >= threshold:
            return date

    # Fallback: use the most frequent date
    most_common = date_counts.most_common(1)[0][0]
    return most_common
