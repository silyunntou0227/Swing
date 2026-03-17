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

    passed = []
    for code in codes:
        stock_prices = prices[prices["Code"] == code]
        if len(stock_prices) < lookback:
            continue

        recent = stock_prices.tail(lookback)

        # 平均出来高チェック
        avg_volume = recent["Volume"].mean()
        if avg_volume < VOLUME_AVG_MIN:
            continue

        # 平均売買代金チェック
        if "TurnoverValue" in recent.columns:
            avg_turnover = recent["TurnoverValue"].mean()
        else:
            # 売買代金 ≈ 終値 × 出来高
            avg_turnover = (recent["Close"] * recent["Volume"]).mean()

        if avg_turnover < TURNOVER_MIN:
            continue

        passed.append(code)

    logger.info(f"流動性フィルタ: {len(codes)} → {len(passed)}銘柄")
    return passed
