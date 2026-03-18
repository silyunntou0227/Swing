"""テクニカルシグナル集約モジュール"""

from __future__ import annotations

import pandas as pd

from src.indicators.trend import (
    calculate_all_trend_indicators,
    detect_golden_dead_cross,
    detect_macd_cross,
    detect_ichimoku_signals,
    detect_granville_signals,
    detect_sma_alignment,
)
from src.indicators.oscillator import (
    calculate_all_oscillators,
    get_oscillator_signals,
)
from src.indicators.volume import (
    calculate_all_volume_indicators,
    get_volume_signals,
)
from src.indicators.pattern import get_pattern_signals
from src.indicators.wave import get_wave_signals
from src.utils.logging_config import logger


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """全テクニカル指標を計算して列に追加

    Args:
        df: 単一銘柄のOHLCVデータ（Date, Open, High, Low, Close, Volume）

    Returns:
        全指標列が追加されたDataFrame
    """
    if len(df) < 30:
        return df

    df = calculate_all_trend_indicators(df)
    df = calculate_all_oscillators(df)
    df = calculate_all_volume_indicators(df)

    return df


def get_all_signals(df: pd.DataFrame) -> list[str]:
    """全テクニカルシグナルを集約して返す

    Args:
        df: 指標計算済みの単一銘柄DataFrame

    Returns:
        検出されたシグナル名のリスト（直近バーの状態）
    """
    if len(df) < 30:
        return []

    signals = []

    # 各シグナル検出を安全に実行するヘルパー
    def _safe_detect(func, name: str) -> list[str]:
        try:
            return func(df)
        except (KeyError, ValueError, IndexError, TypeError) as e:
            logger.debug(f"シグナル検出エラー ({name}): {e}")
            return []

    # トレンド系シグナル
    signals.extend(_safe_detect(detect_golden_dead_cross, "GC/DC"))
    signals.extend(_safe_detect(detect_macd_cross, "MACD"))
    signals.extend(_safe_detect(detect_ichimoku_signals, "一目均衡表"))
    signals.extend(_safe_detect(detect_sma_alignment, "SMA配列"))
    signals.extend(_safe_detect(detect_granville_signals, "グランビル"))

    # オシレーター系シグナル
    signals.extend(_safe_detect(get_oscillator_signals, "オシレーター"))

    # 出来高系シグナル
    signals.extend(_safe_detect(get_volume_signals, "出来高"))

    # パターン系シグナル
    signals.extend(_safe_detect(get_pattern_signals, "パターン"))

    # 波動分析シグナル
    signals.extend(_safe_detect(get_wave_signals, "波動"))

    return signals


def count_buy_sell_signals(signals: list[str]) -> tuple[int, int]:
    """シグナルリストから買い/売りシグナルの数をカウント

    Returns:
        (buy_count, sell_count)
    """
    buy_keywords = [
        "GC", "ゴールデンクロス", "買い", "上昇", "反転上昇",
        "雲上抜け", "三役好転", "RSI反転上昇", "包み足(強気)",
        "サポート", "第3波上昇", "出来高急増",
    ]
    sell_keywords = [
        "DC", "デッドクロス", "売り", "下降", "反転下降",
        "雲下抜け", "三役逆転", "RSI反転下降", "包み足(弱気)",
        "レジスタンス", "天井", "A波下落",
    ]

    buy_count = 0
    sell_count = 0
    for signal in signals:
        if any(kw in signal for kw in buy_keywords):
            buy_count += 1
        if any(kw in signal for kw in sell_keywords):
            sell_count += 1

    return buy_count, sell_count
