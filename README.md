# P2 ETF Change-Point Adaptive Engine

**Regime‑aware ETF selection using Bayesian Changepoint Detection (PELT) and adaptive training windows.**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-CHANGE-POINT-ADAPTIVE/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-CHANGE-POINT-ADAPTIVE/actions/workflows/daily_run.yml)

## Overview

This engine detects structural breaks (regime shifts) in ETF return dynamics and dynamically adjusts the training window to use only post‑change data. By ignoring obsolete regimes, the model stays responsive to current market conditions.

**Key Features:**
- **Change Point Detection**: Uses the efficient PELT algorithm (`ruptures` library) on daily log returns.
- **Adaptive Lookback**: Training window automatically resets after a consensus change point across the universe.
- **Three Universes**: FI/Commodities, Equity Sectors, and Combined.
- **Lightweight Model**: Ridge regression with cross‑validation for fast, robust predictions.
- **Daily Output**: Top ETF pick, change point date, adaptive lookback length, and performance metrics.

## Data

- **Input**: `P2SAMAPA/fi-etf-macro-signal-master-data` (master_data.parquet)
- **Output**: `P2SAMAPA/p2-etf-change-point-adaptive-results`

## Usage

```bash
pip install -r requirements.txt
python adaptive_trainer.py   # Runs training and pushes to HF
streamlit run streamlit_app.py
Configuration
All parameters are in config.py:

Change point sensitivity (CP_PENALTY, CP_MIN_DAYS_BETWEEN)

Consensus fraction for universe‑wide change point (CP_CONSENSUS_FRACTION)

Model hyperparameters (LOOKBACK_FEATURES, MODEL_ALPHA)
