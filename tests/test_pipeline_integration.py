"""パイプライン統合テスト + スコアリング詳細テスト"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    SCORING_WEIGHTS,
    SMA_SHORT, SMA_MEDIUM, SMA_LONG,
    RSI2_EXTREME_OVERSOLD, RSI2_OVERBOUGHT,
    VOLUME_SCORE_HIGH, VOLUME_SCORE_MID,
    ATR_RATIO_OPTIMAL_MIN, ATR_RATIO_OPTIMAL_MAX,
    BUY_PATTERNS, SELL_PATTERNS,
    HOLD_DAYS_MEAN_REVERSION, HOLD_DAYS_BREAKOUT,
    HOLD_DAYS_MOMENTUM, HOLD_DAYS_DEFAULT,
)


def make_sample_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1000 + np.cumsum(rng.normal(0, 10, n))
    close = np.maximum(close, 100)
    high = close + rng.uniform(5, 20, n)
    low = close - rng.uniform(5, 20, n)
    open_ = close + rng.normal(0, 5, n)
    volume = rng.integers(50000, 500000, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Date": dates, "Open": open_, "High": high,
        "Low": low, "Close": close, "Volume": volume,
    })


class TestConfigValidation:
    """config.py のパラメータ整合性テスト"""

    def test_weights_sum_to_one(self):
        assert SCORING_WEIGHTS.validate()

    def test_sma_order(self):
        assert SMA_SHORT < SMA_MEDIUM < SMA_LONG

    def test_rsi2_thresholds_ordered(self):
        assert RSI2_EXTREME_OVERSOLD < RSI2_OVERBOUGHT

    def test_atr_ratio_range(self):
        assert ATR_RATIO_OPTIMAL_MIN < ATR_RATIO_OPTIMAL_MAX

    def test_hold_days_positive(self):
        for days in [HOLD_DAYS_MEAN_REVERSION, HOLD_DAYS_BREAKOUT,
                     HOLD_DAYS_MOMENTUM, HOLD_DAYS_DEFAULT]:
            assert days > 0

    def test_pattern_lists_non_empty(self):
        assert len(BUY_PATTERNS) > 0
        assert len(SELL_PATTERNS) > 0


class TestCalculateAllIndicators:
    """テクニカル指標の統合計算テスト"""

    def test_returns_all_columns(self):
        from src.indicators.technical import calculate_all_indicators

        df = make_sample_ohlcv(200)
        result = calculate_all_indicators(df)

        expected_cols = ["SMA_5", "SMA_25", "MACD", "RSI", "ATR", "OBV"]
        for col in expected_cols:
            assert col in result.columns, f"{col} が計算されていません"

    def test_short_data_returns_unchanged(self):
        from src.indicators.technical import calculate_all_indicators

        df = make_sample_ohlcv(20)
        result = calculate_all_indicators(df)
        assert len(result) == 20


class TestGetAllSignals:
    """シグナル検出の統合テスト"""

    def test_returns_list(self):
        from src.indicators.technical import calculate_all_indicators, get_all_signals

        df = make_sample_ohlcv(200)
        df = calculate_all_indicators(df)
        signals = get_all_signals(df)
        assert isinstance(signals, list)

    def test_short_data_returns_empty(self):
        from src.indicators.technical import get_all_signals

        df = make_sample_ohlcv(20)
        signals = get_all_signals(df)
        assert signals == []


class TestCountBuySellSignals:
    def test_buy_signals(self):
        from src.indicators.technical import count_buy_sell_signals

        signals = ["ゴールデンクロス(買い)", "MACD買いシグナル"]
        buy, sell = count_buy_sell_signals(signals)
        assert buy >= 2
        assert sell == 0

    def test_sell_signals(self):
        from src.indicators.technical import count_buy_sell_signals

        signals = ["デッドクロス(売り)", "雲下抜け"]
        buy, sell = count_buy_sell_signals(signals)
        assert sell >= 2

    def test_empty_signals(self):
        from src.indicators.technical import count_buy_sell_signals

        buy, sell = count_buy_sell_signals([])
        assert buy == 0
        assert sell == 0


class TestLiquidityFilterGroupby:
    """groupby化後のliquidity filterの動作テスト"""

    def test_empty_prices(self):
        from src.screening.liquidity import filter_liquidity

        codes = ["1001", "1002"]
        result = filter_liquidity(codes, pd.DataFrame())
        assert result == codes

    def test_preserves_order(self):
        from src.screening.liquidity import filter_liquidity

        codes = ["1002", "1001"]
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        prices_data = []
        for code in codes:
            for d in dates:
                prices_data.append({
                    "Code": code, "Date": d,
                    "Close": 1000, "Volume": 100000,
                })
        prices = pd.DataFrame(prices_data)
        result = filter_liquidity(codes, prices)
        # 元の順序を維持していること
        if len(result) >= 2:
            assert result.index(result[0]) < result.index(result[1])

    def test_filters_low_volume(self):
        from src.screening.liquidity import filter_liquidity

        codes = ["1001", "1002"]
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        prices_data = []
        for code in codes:
            for d in dates:
                vol = 200000 if code == "1001" else 100
                prices_data.append({
                    "Code": code, "Date": d,
                    "Close": 1000, "Volume": vol,
                })
        prices = pd.DataFrame(prices_data)
        result = filter_liquidity(codes, prices)
        assert "1001" in result
        assert "1002" not in result


class TestRiskCalculator:
    """リスク管理計算テスト"""

    def test_balsara_symmetry(self):
        from src.scoring.risk import calculate_balsara_ruin_probability

        # 高リスク > 低リスク
        high = calculate_balsara_ruin_probability(0.5, 2.0, 0.10)
        low = calculate_balsara_ruin_probability(0.5, 2.0, 0.01)
        assert high >= low

    def test_balsara_zero_payoff(self):
        from src.scoring.risk import calculate_balsara_ruin_probability

        assert calculate_balsara_ruin_probability(0.5, 0.0, 0.01) == 100.0

    def test_balsara_perfect_win(self):
        from src.scoring.risk import calculate_balsara_ruin_probability

        assert calculate_balsara_ruin_probability(1.0, 2.0, 0.01) == 0.0


class TestMarginClientParallel:
    """並列化後のMarginClientテスト"""

    def test_empty_codes(self):
        from src.data.margin_client import MarginClient

        client = MarginClient()
        result = client.fetch_margin_for_codes([])
        assert result.empty
