"""流動性フィルタモジュール"""

from __future__ import annotations

import pandas as pd

from src.config import VOLUME_AVG_MIN, TURNOVER_MIN
from src.utils.logging_config import logger


def filter_liquidity(
    codes: list[str],
    prices: pd.DataFrame,
    lookback: int = 20,
) -> list[str]:
    """流動性フィルタを適用

    Args:
        codes: 銘柄コードリスト
        prices: 全銘柄の株価データ
        lookback: 平均出来高の計算期間

    Returns:
        フィルタ通過した銘柄コードリスト
    """
    if prices.empty:
        return codes

    # 対象銘柄のデータのみ抽出（1回のフィルタリング）
    target_prices = prices[prices["Code"].isin(codes)]
    if target_prices.empty:
        logger.info(f"流動性フィルタ: {len(codes)} → 0銘柄（価格データなし）")
        return []

    has_turnover = "TurnoverValue" in target_prices.columns

    # groupby で一括処理（N+1 → 1回の走査）
    passed_codes = []
    for code, group in target_prices.groupby("Code"):
        if len(group) < lookback:
            continue
        recent = group.tail(lookback)
        if recent["Volume"].mean() < VOLUME_AVG_MIN:
            continue
        if has_turnover:
            avg_turnover = recent["TurnoverValue"].mean()
        else:
            avg_turnover = (recent["Close"] * recent["Volume"]).mean()
        if avg_turnover >= TURNOVER_MIN:
            passed_codes.append(code)

    # 元の codes の順序を維持
    codes_set = set(passed_codes)
    passed = [c for c in codes if c in codes_set]

    logger.info(f"流動性フィルタ: {len(codes)} → {len(passed)}銘柄")
    return passed
