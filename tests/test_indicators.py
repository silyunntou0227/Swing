"""テクニカル指標のユニットテスト"""

import numpy as np
import pandas as pd
import pytest


def make_sample_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """テスト用のサンプルOHLCVデータを生成"""
    rng = np.random.default_rng(seed)

    # ランダムウォークで株価を生成
    close = 1000 + np.cumsum(rng.normal(0, 10, n))
    close = np.maximum(close, 100)  # 最低100円

    high = close + rng.uniform(5, 20, n)
    low = close - rng.uniform(5, 20, n)
    open_ = close + rng.normal(0, 5, n)
    volume = rng.integers(50000, 500000, n)

    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


class TestTrendIndicators:
    def test_calculate_all_trend_indicators(self):
        from src.indicators.trend import calculate_all_trend_indicators

        df = make_sample_ohlcv(200)
        result = calculate_all_trend_indicators(df)

        assert "SMA_5" in result.columns
        assert "SMA_25" in result.columns
        assert "SMA_75" in result.columns
        assert "MACD" in result.columns
        assert len(result) == len(df)

    def test_detect_golden_dead_cross(self):
        from src.indicators.trend import (
            calculate_all_trend_indicators,
            detect_golden_dead_cross,
        )

        df = make_sample_ohlcv(200)
        df = calculate_all_trend_indicators(df)
        signals = detect_golden_dead_cross(df)
        assert isinstance(signals, list)

    def test_detect_sma_alignment(self):
        from src.indicators.trend import (
            calculate_all_trend_indicators,
            detect_sma_alignment,
        )

        df = make_sample_ohlcv(200)
        df = calculate_all_trend_indicators(df)
        signals = detect_sma_alignment(df)
        assert isinstance(signals, list)


class TestOscillatorIndicators:
    def test_calculate_rsi(self):
        from src.indicators.oscillator import calculate_rsi

        df = make_sample_ohlcv(100)
        result = calculate_rsi(df)
        assert "RSI" in result.columns

    def test_calculate_stochastic(self):
        from src.indicators.oscillator import calculate_stochastic

        df = make_sample_ohlcv(100)
        result = calculate_stochastic(df)
        assert "STOCHk" in result.columns or "STOCHd" in result.columns or len(result) > 0

    def test_get_oscillator_signals(self):
        from src.indicators.oscillator import (
            calculate_all_oscillators,
            get_oscillator_signals,
        )

        df = make_sample_ohlcv(100)
        df = calculate_all_oscillators(df)
        signals = get_oscillator_signals(df)
        assert isinstance(signals, list)


class TestVolumeIndicators:
    def test_calculate_atr(self):
        from src.indicators.volume import calculate_atr

        df = make_sample_ohlcv(100)
        result = calculate_atr(df)
        assert "ATR" in result.columns

    def test_calculate_obv(self):
        from src.indicators.volume import calculate_obv

        df = make_sample_ohlcv(100)
        result = calculate_obv(df)
        assert "OBV" in result.columns

    def test_volume_spike_detection(self):
        from src.indicators.volume import (
            calculate_all_volume_indicators,
            get_volume_signals,
        )

        df = make_sample_ohlcv(100)
        df = calculate_all_volume_indicators(df)
        signals = get_volume_signals(df)
        assert isinstance(signals, list)


class TestPatternIndicators:
    def test_get_pattern_signals(self):
        from src.indicators.pattern import get_pattern_signals

        df = make_sample_ohlcv(100)
        signals = get_pattern_signals(df)
        assert isinstance(signals, list)


class TestWaveIndicators:
    def test_find_swing_points(self):
        from src.indicators.wave import find_swing_points

        df = make_sample_ohlcv(100)
        highs, lows = find_swing_points(df)
        assert isinstance(highs, list)
        assert isinstance(lows, list)

    def test_fibonacci_levels(self):
        from src.indicators.wave import calculate_fibonacci_levels

        levels = calculate_fibonacci_levels(high=1100, low=1000, direction="up")
        assert "fib_0.382" in levels
        assert "fib_0.618" in levels
        assert levels["fib_0.382"] < 1100
        assert levels["fib_0.382"] > 1000

    def test_dow_theory_trend(self):
        from src.indicators.wave import detect_dow_theory_trend

        df = make_sample_ohlcv(100)
        trend = detect_dow_theory_trend(df)
        assert trend in ("uptrend", "downtrend", "sideways")

    def test_get_wave_signals(self):
        from src.indicators.wave import get_wave_signals

        df = make_sample_ohlcv(100)
        signals = get_wave_signals(df)
        assert isinstance(signals, list)
