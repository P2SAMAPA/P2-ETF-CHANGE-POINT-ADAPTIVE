"""
Configuration for P2-ETF-CHANGE-POINT-ADAPTIVE.
"""
import os

# Hugging Face configuration
HF_INPUT_DATASET = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_INPUT_FILE = "master_data.parquet"
HF_OUTPUT_DATASET = "P2SAMAPA/p2-etf-change-point-adaptive-results"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Universes
FI_COMMODITY_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_TICKERS = ["QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "GDX", "XME"]
COMBINED_TICKERS = FI_COMMODITY_TICKERS + EQUITY_TICKERS

BENCHMARK_FI = "AGG"
BENCHMARK_EQ = "SPY"

# Macro columns
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M", "IG_SPREAD", "HY_SPREAD"]

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_TRAIN_DAYS = 252 * 2
TRADING_DAYS_PER_YEAR = 252

# Change Point Detection parameters (ruptures.Pelt)
CP_PENALTY = 3.0               # Higher = fewer change points
CP_MODEL = "l2"                # Detects shifts in mean
CP_MIN_DAYS_BETWEEN = 20       # Minimum days between change points
CP_PROB_THRESHOLD = 0.7        # Not used with PELT; kept for interface consistency

# Adaptive window: use change point if at least this fraction of ETFs agree
CP_CONSENSUS_FRACTION = 0.5

# Model parameters
MODEL_ALPHA = 1.0              # Ridge regularization (tuned by RidgeCV)
LOOKBACK_FEATURES = [1, 2, 3, 5, 10, 21]  # Lagged returns as features
