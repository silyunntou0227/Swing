"""出来高分析モジュール

出来高スパイク検出、OBV、ATR、出来高トレンド分析を提供する。
DataFrameには Date, Open, High, Low, Close, Volume カラムが必要。
"""

from __future__ import annotations

from typing import List

import pandas as pd
import ta as ta_lib

from src.config import ATR_PERIOD, VOLUME_SPIKE_RATIO


# ------------------------------------------------------------------
# ATR（Average True Range）
# ------------------------------------------------------------------
def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """ATR を計算して 'ATR' カラムを追加する。"""
    df = df.copy()
    df["ATR"] = ta_lib.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=period,
    ).average_true_range()
    return df


# ------------------------------------------------------------------
# OBV（On Balance Volume）
# ------------------------------------------------------------------
def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """OBV を計算して 'OBV' カラムを追加する。"""
    df = df.copy()
    close_diff = df["Close"].diff()
    sign = close_diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["OBV"] = (sign * df["Volume"]).cumsum()
    return df


# ------------------------------------------------------------------
# 出来高比率（Volume Ratio）
# ------------------------------------------------------------------
def calculate_volume_ratio(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    """当日出来高 / N日平均出来高 の比率を計算する。"""
    df = df.copy()
    avg_volume = df["Volume"].rolling(window=period).mean()
    df["VolumeRatio"] = df["Volume"] / avg_volume
    return df


# ------------------------------------------------------------------
# 出来高スパイク検出
# ------------------------------------------------------------------
def detect_volume_spike(df: pd.DataFrame) -> pd.DataFrame:
    """出来高が 20日平均 * VOLUME_SPIKE_RATIO を超えた日を検出する。"""
    df = df.copy()
    if "VolumeRatio" not in df.columns:
        df = calculate_volume_ratio(df)
    df["VolumeSpike"] = df["VolumeRatio"] >= VOLUME_SPIKE_RATIO
    return df


# ------------------------------------------------------------------
# 出来高トレンド分析
# ------------------------------------------------------------------
def _volume_trend(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """直近 *window* 日間の出来高が増加傾向か減少傾向かを判定する。"""
    slopes: List[str] = []
    volumes = df["Volume"].values
    for i in range(len(volumes)):
        if i < window - 1:
            slopes.append("flat")
            continue
        segment = volumes[i - window + 1: i + 1]
        x_mean = (window - 1) / 2.0
        y_mean = segment.mean()
        numerator = sum((j - x_mean) * (v - y_mean) for j, v in enumerate(segment))
        denominator = sum((j - x_mean) ** 2 for j in range(window))
        slope = numerator / denominator if denominator != 0 else 0.0
        if slope > 0:
            slopes.append("increasing")
        elif slope < 0:
            slopes.append("decreasing")
        else:
            slopes.append("flat")
    return pd.Series(slopes, index=df.index)


# ------------------------------------------------------------------
# 全出来高指標を一括計算
# ------------------------------------------------------------------
def calculate_all_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """全ての出来高関連指標を一括で追加する。"""
    df = calculate_atr(df)
    df = calculate_obv(df)
    df = calculate_volume_ratio(df)
    df = detect_volume_spike(df)
    df["VolumeTrend"] = _volume_trend(df)
    return df


# ------------------------------------------------------------------
# 直近バーの出来高シグナル取得
# ------------------------------------------------------------------
def get_volume_signals(df: pd.DataFrame) -> List[str]:
    """直近（最新）バーの出来高関連シグナルを文字列リストで返す。"""
    df = calculate_all_volume_indicators(df)
    if df.empty:
        return []

    signals: List[str] = []
    last = df.iloc[-1]

    if last.get("VolumeSpike", False):
        ratio = last.get("VolumeRatio", 0.0)
        signals.append(f"出来高スパイク ({ratio:.1f}倍)")

    trend = last.get("VolumeTrend", "flat")
    if trend != "flat":
        signals.append(f"出来高トレンド: {trend}")

    if len(df) >= 2:
        obv_now = last.get("OBV", 0)
        obv_prev = df.iloc[-2].get("OBV", 0)
        if obv_now > obv_prev:
            signals.append("OBV上昇")
        elif obv_now < obv_prev:
            signals.append("OBV下降")

    return signals
