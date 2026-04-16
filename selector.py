"""
ETF selection logic.
"""
def select_top_etf(predictions: dict) -> str:
    """Return the ticker with the highest predicted return."""
    if not predictions:
        return None
    return max(predictions, key=predictions.get)
