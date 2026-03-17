"""リスク管理・ポジションサイジングモジュール"""

from __future__ import annotations

import math

import pandas as pd

from src.config import (
    RISK_PER_TRADE,
    DEFAULT_CAPITAL,
    STOP_LOSS_ATR_MULTIPLIER,
    TAKE_PROFIT_RR_RATIO,
    BALSARA_WIN_RATE,
    BALSARA_PAYOFF_RATIO,
    ATR_PERIOD,
)
from src.data.data_loader import MarketData
from src.notify.formatter import ScoredCandidate
from src.utils.logging_config import logger


class RiskCalculator:
    """リスク管理計算"""

    def __init__(self, capital: float = DEFAULT_CAPITAL) -> None:
        self._capital = capital

    def calculate(
        self, candidate: ScoredCandidate, market_data: MarketData
    ) -> None:
        """候補銘柄のリスク管理指標を計算してScoredCandidateに設定

        Args:
            candidate: スコアリング済み候補（in-place更新）
            market_data: 全データ
        """
        close = candidate.close
        if close <= 0:
            return

        # 株価データからATRを取得
        atr = self._get_atr(candidate.code, market_data)
        if atr is None or atr <= 0:
            return

        # 損切りライン
        if candidate.direction == "buy":
            candidate.stop_loss = round(close - atr * STOP_LOSS_ATR_MULTIPLIER, 1)
            candidate.take_profit = round(
                close + atr * STOP_LOSS_ATR_MULTIPLIER * TAKE_PROFIT_RR_RATIO, 1
            )
        else:
            candidate.stop_loss = round(close + atr * STOP_LOSS_ATR_MULTIPLIER, 1)
            candidate.take_profit = round(
                close - atr * STOP_LOSS_ATR_MULTIPLIER * TAKE_PROFIT_RR_RATIO, 1
            )

        # リスク/リワード比率
        risk = abs(close - candidate.stop_loss)
        reward = abs(candidate.take_profit - close)
        candidate.risk_reward_ratio = round(reward / risk, 1) if risk > 0 else 0.0

        # ポジションサイズ（1%ルール）
        risk_amount = self._capital * RISK_PER_TRADE
        if risk > 0:
            shares = int(risk_amount / risk)
            # 単元株（100株単位）に丸め
            shares = max(100, (shares // 100) * 100)
            candidate.position_size_shares = shares
            candidate.position_size_yen = round(shares * close)

        # バルサラの破産確率
        candidate.ruin_probability = calculate_balsara_ruin_probability(
            win_rate=BALSARA_WIN_RATE,
            payoff_ratio=candidate.risk_reward_ratio or BALSARA_PAYOFF_RATIO,
            risk_per_trade=RISK_PER_TRADE,
        )

    def _get_atr(self, code: str, market_data: MarketData) -> float | None:
        """株価データからATRを取得"""
        if market_data.prices.empty:
            return None

        stock_prices = market_data.prices[market_data.prices["Code"] == code]
        if stock_prices.empty:
            return None

        # ATRが計算済みならそれを使用
        if "ATR" in stock_prices.columns:
            atr_val = stock_prices["ATR"].iloc[-1]
            if pd.notna(atr_val):
                return float(atr_val)

        # 手動計算
        if len(stock_prices) < ATR_PERIOD + 1:
            return None

        recent = stock_prices.tail(ATR_PERIOD + 1).copy()
        high = recent["High"].values
        low = recent["Low"].values
        close = recent["Close"].values

        true_ranges = []
        for i in range(1, len(recent)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            true_ranges.append(tr)

        if not true_ranges:
            return None

        return sum(true_ranges) / len(true_ranges)


def calculate_balsara_ruin_probability(
    win_rate: float = BALSARA_WIN_RATE,
    payoff_ratio: float = BALSARA_PAYOFF_RATIO,
    risk_per_trade: float = RISK_PER_TRADE,
) -> float:
    """バルサラの破産確率を計算

    Args:
        win_rate: 勝率（0-1）
        payoff_ratio: 損益比（利益/損失）
        risk_per_trade: 1トレードあたりのリスク（資金比、0-1）

    Returns:
        破産確率（0-100%）
    """
    if win_rate <= 0 or win_rate >= 1:
        return 100.0 if win_rate <= 0 else 0.0

    if payoff_ratio <= 0 or risk_per_trade <= 0:
        return 100.0

    loss_rate = 1 - win_rate

    # 期待値
    edge = win_rate * payoff_ratio - loss_rate
    if edge <= 0:
        return 100.0  # 期待値がマイナス → 破産確実

    # バルサラの近似式
    # P(ruin) ≈ (q / (p * R))^(1/f)
    # p=勝率, q=敗率, R=損益比, f=リスク比率
    try:
        base = loss_rate / (win_rate * payoff_ratio)
        if base >= 1:
            return 100.0
        exponent = 1.0 / risk_per_trade
        ruin_prob = base ** exponent
        return round(max(0, min(100, ruin_prob * 100)), 2)
    except (ZeroDivisionError, OverflowError, ValueError):
        return 100.0
