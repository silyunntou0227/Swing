"""トレンド系テクニカル指標モジュール

移動平均線、MACD、一目均衡表、ADX などのトレンド系指標の算出と
ゴールデンクロス／デッドクロス、グランビルの法則等のシグナル検出を行う。

対象DataFrame列: Date, Open, High, Low, Close, Volume
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

try:
    import ta as ta_lib
except ImportError:
    ta_lib = None  # type: ignore[assignment]

from src.config import (
    ADX_MIN,
    EMA_LONG,
    EMA_SHORT,
    ICHIMOKU_KIJUN,
    ICHIMOKU_SENKOU_B,
    ICHIMOKU_TENKAN,
    MACD_SIGNAL,
    SMA_LONG,
    SMA_MEDIUM,
    SMA_SHORT,
    SMA_VERY_LONG,
)

logger = logging.getLogger(__name__)


# ============================================================
# 移動平均線（SMA / EMA）
# ============================================================

def calc_sma(df: pd.DataFrame) -> pd.DataFrame:
    """SMA (5, 25, 75, 200) を算出して返す。"""
    result = pd.DataFrame(index=df.index)
    for period, label in [
        (SMA_SHORT, f"SMA_{SMA_SHORT}"),
        (SMA_MEDIUM, f"SMA_{SMA_MEDIUM}"),
        (SMA_LONG, f"SMA_{SMA_LONG}"),
        (SMA_VERY_LONG, f"SMA_{SMA_VERY_LONG}"),
    ]:
        result[label] = ta_lib.trend.sma_indicator(df["Close"], window=period)
    return result


def calc_ema(df: pd.DataFrame) -> pd.DataFrame:
    """EMA (12, 26) を算出して返す。"""
    result = pd.DataFrame(index=df.index)
    for period, label in [
        (EMA_SHORT, f"EMA_{EMA_SHORT}"),
        (EMA_LONG, f"EMA_{EMA_LONG}"),
    ]:
        result[label] = ta_lib.trend.ema_indicator(df["Close"], window=period)
    return result


# ============================================================
# MACD
# ============================================================

def calc_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD (12, 26, 9) を算出して返す。"""
    macd_indicator = ta_lib.trend.MACD(
        df["Close"],
        window_slow=EMA_LONG,
        window_fast=EMA_SHORT,
        window_sign=MACD_SIGNAL,
    )
    result = pd.DataFrame(index=df.index)
    result["MACD"] = macd_indicator.macd()
    result["MACD_Signal"] = macd_indicator.macd_signal()
    result["MACD_Hist"] = macd_indicator.macd_diff()
    return result


# ============================================================
# 一目均衡表
# ============================================================

def calc_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """一目均衡表の各線を算出して返す。"""
    result = pd.DataFrame(index=df.index)

    # 転換線: (9日間の最高値 + 9日間の最安値) / 2
    high_t = df["High"].rolling(window=ICHIMOKU_TENKAN).max()
    low_t = df["Low"].rolling(window=ICHIMOKU_TENKAN).min()
    result["Ichimoku_Tenkan"] = (high_t + low_t) / 2

    # 基準線: (26日間の最高値 + 26日間の最安値) / 2
    high_k = df["High"].rolling(window=ICHIMOKU_KIJUN).max()
    low_k = df["Low"].rolling(window=ICHIMOKU_KIJUN).min()
    result["Ichimoku_Kijun"] = (high_k + low_k) / 2

    # 先行スパンA: (転換線 + 基準線) / 2 を26日先行
    result["Ichimoku_SenkouA"] = (
        (result["Ichimoku_Tenkan"] + result["Ichimoku_Kijun"]) / 2
    ).shift(ICHIMOKU_KIJUN)

    # 先行スパンB: (52日間の最高値 + 52日間の最安値) / 2 を26日先行
    high_b = df["High"].rolling(window=ICHIMOKU_SENKOU_B).max()
    low_b = df["Low"].rolling(window=ICHIMOKU_SENKOU_B).min()
    result["Ichimoku_SenkouB"] = ((high_b + low_b) / 2).shift(ICHIMOKU_KIJUN)

    # 遅行スパン: 当日終値を26日遅行
    result["Ichimoku_Chikou"] = df["Close"].shift(-ICHIMOKU_KIJUN)

    return result


# ============================================================
# ADX (Average Directional Index)
# ============================================================

def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX, +DI, -DI を算出して返す。"""
    adx_indicator = ta_lib.trend.ADXIndicator(
        df["High"], df["Low"], df["Close"], window=period
    )
    result = pd.DataFrame(index=df.index)
    result["ADX"] = adx_indicator.adx()
    result["ADX_Plus_DI"] = adx_indicator.adx_pos()
    result["ADX_Minus_DI"] = adx_indicator.adx_neg()
    return result


# ============================================================
# 全トレンド指標一括算出
# ============================================================

def calculate_all_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """全トレンド系テクニカル指標をDataFrameに追加して返す。"""
    result = df.copy()

    for sub_df in [calc_sma(result), calc_ema(result), calc_macd(result),
                   calc_ichimoku(result), calc_adx(result)]:
        for col in sub_df.columns:
            result[col] = sub_df[col]

    result["SMA_Perfect_Order_Bull"] = _is_perfect_order_bull(result)
    result["SMA_Perfect_Order_Bear"] = _is_perfect_order_bear(result)
    return result


# ============================================================
# SMAアラインメント (パーフェクトオーダー)
# ============================================================

def _is_perfect_order_bull(df: pd.DataFrame) -> pd.Series:
    s5, s25, s75, s200 = (f"SMA_{p}" for p in [SMA_SHORT, SMA_MEDIUM, SMA_LONG, SMA_VERY_LONG])
    return (df[s5] > df[s25]) & (df[s25] > df[s75]) & (df[s75] > df[s200])


def _is_perfect_order_bear(df: pd.DataFrame) -> pd.Series:
    s5, s25, s75, s200 = (f"SMA_{p}" for p in [SMA_SHORT, SMA_MEDIUM, SMA_LONG, SMA_VERY_LONG])
    return (df[s5] < df[s25]) & (df[s25] < df[s75]) & (df[s75] < df[s200])


# ============================================================
# シグナル検出: ゴールデンクロス / デッドクロス
# ============================================================

def detect_golden_dead_cross(df: pd.DataFrame) -> List[str]:
    """SMA5 と SMA25 のゴールデンクロス / デッドクロスを検出する。"""
    signals: List[str] = []
    s5 = f"SMA_{SMA_SHORT}"
    s25 = f"SMA_{SMA_MEDIUM}"

    if len(df) < 2 or s5 not in df.columns or s25 not in df.columns:
        return signals

    prev, curr = df.iloc[-2], df.iloc[-1]

    if prev[s5] <= prev[s25] and curr[s5] > curr[s25]:
        signals.append("SMA_GoldenCross_5_25")
    if prev[s5] >= prev[s25] and curr[s5] < curr[s25]:
        signals.append("SMA_DeadCross_5_25")

    return signals


# ============================================================
# シグナル検出: MACDクロス
# ============================================================

def detect_macd_cross(df: pd.DataFrame) -> List[str]:
    """MACDラインとシグナルラインのクロスを検出する。"""
    signals: List[str] = []
    if len(df) < 2:
        return signals
    for col in ("MACD", "MACD_Signal"):
        if col not in df.columns:
            return signals

    prev, curr = df.iloc[-2], df.iloc[-1]

    if prev["MACD"] <= prev["MACD_Signal"] and curr["MACD"] > curr["MACD_Signal"]:
        signals.append("MACD_BullishCross_BelowZero" if curr["MACD"] < 0 else "MACD_BullishCross_AboveZero")
    if prev["MACD"] >= prev["MACD_Signal"] and curr["MACD"] < curr["MACD_Signal"]:
        signals.append("MACD_BearishCross_AboveZero" if curr["MACD"] > 0 else "MACD_BearishCross_BelowZero")
    if prev["MACD"] <= 0 and curr["MACD"] > 0:
        signals.append("MACD_ZeroCross_Bullish")
    if prev["MACD"] >= 0 and curr["MACD"] < 0:
        signals.append("MACD_ZeroCross_Bearish")

    return signals


# ============================================================
# シグナル検出: 一目均衡表
# ============================================================

def detect_ichimoku_signals(df: pd.DataFrame) -> List[str]:
    """一目均衡表のシグナルを検出する。"""
    signals: List[str] = []
    required = ["Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_SenkouA", "Ichimoku_SenkouB"]
    if len(df) < 2 or not all(col in df.columns for col in required):
        return signals

    prev, curr = df.iloc[-2], df.iloc[-1]

    cloud_top = max(curr["Ichimoku_SenkouA"], curr["Ichimoku_SenkouB"])
    cloud_bottom = min(curr["Ichimoku_SenkouA"], curr["Ichimoku_SenkouB"])
    prev_cloud_top = max(prev["Ichimoku_SenkouA"], prev["Ichimoku_SenkouB"])
    prev_cloud_bottom = min(prev["Ichimoku_SenkouA"], prev["Ichimoku_SenkouB"])

    if prev["Close"] <= prev_cloud_top and curr["Close"] > cloud_top:
        signals.append("Ichimoku_CloudBreakout_Bullish")
    if prev["Close"] >= prev_cloud_bottom and curr["Close"] < cloud_bottom:
        signals.append("Ichimoku_CloudBreakout_Bearish")

    if prev["Ichimoku_Tenkan"] <= prev["Ichimoku_Kijun"] and curr["Ichimoku_Tenkan"] > curr["Ichimoku_Kijun"]:
        signals.append("Ichimoku_TenkanKijun_BullishCross")
    if prev["Ichimoku_Tenkan"] >= prev["Ichimoku_Kijun"] and curr["Ichimoku_Tenkan"] < curr["Ichimoku_Kijun"]:
        signals.append("Ichimoku_TenkanKijun_BearishCross")

    # 三役好転
    is_above = curr["Ichimoku_Tenkan"] > curr["Ichimoku_Kijun"] and curr["Close"] > cloud_top
    chikou_ok = False
    if "Ichimoku_Chikou" in df.columns and len(df) > ICHIMOKU_KIJUN:
        idx = len(df) - 1 - ICHIMOKU_KIJUN
        if idx >= 0:
            cv = df.iloc[-1]["Ichimoku_Chikou"]
            pc = df.iloc[idx]["Close"]
            if pd.notna(cv) and pd.notna(pc):
                chikou_ok = cv > pc
    if is_above and chikou_ok:
        signals.append("Ichimoku_Sannyaku_Kouten")

    # 三役逆転
    is_below = curr["Ichimoku_Tenkan"] < curr["Ichimoku_Kijun"] and curr["Close"] < cloud_bottom
    chikou_bear = False
    if "Ichimoku_Chikou" in df.columns and len(df) > ICHIMOKU_KIJUN:
        idx = len(df) - 1 - ICHIMOKU_KIJUN
        if idx >= 0:
            cv = df.iloc[-1]["Ichimoku_Chikou"]
            pc = df.iloc[idx]["Close"]
            if pd.notna(cv) and pd.notna(pc):
                chikou_bear = cv < pc
    if is_below and chikou_bear:
        signals.append("Ichimoku_Sannyaku_Gyakuten")

    return signals


# ============================================================
# シグナル検出: グランビルの法則
# ============================================================

def detect_granville_signals(
    df: pd.DataFrame,
    sma_col: str = f"SMA_{SMA_MEDIUM}",
    close_col: str = "Close",
) -> List[str]:
    """グランビルの法則（8つのルール）に基づくシグナルを検出する。"""
    signals: List[str] = []
    if sma_col not in df.columns or len(df) < 4:
        return signals

    d = df[[close_col, sma_col]].iloc[-4:].copy()
    d["SMA_Diff"] = d[sma_col].diff()
    curr, prev, prev2 = d.iloc[-1], d.iloc[-2], d.iloc[-3]

    sma_rising = curr["SMA_Diff"] > 0
    sma_falling = curr["SMA_Diff"] < 0
    cc, cp, cp2 = curr[close_col], prev[close_col], prev2[close_col]
    sc, sp, sp2 = curr[sma_col], prev[sma_col], prev2[sma_col]
    deviation = (cc - sc) / sc if sc != 0 else 0

    if sma_rising and cp <= sp and cc > sc:
        signals.append("Granville_Buy1_CrossAbove")
    if sma_rising and cp2 > sp2 and cp < sp and cc > sc:
        signals.append("Granville_Buy2_BounceBack")
    if sma_rising and cp > sp and cp2 > sp2 and cp < cp2 and cc > cp and abs(cp - sp) / sp < 0.02:
        signals.append("Granville_Buy3_NearSMA")
    if sma_falling and deviation < -0.10:
        signals.append("Granville_Buy4_Oversold")
    if sma_falling and cp >= sp and cc < sc:
        signals.append("Granville_Sell5_CrossBelow")
    if sma_falling and cp2 < sp2 and cp > sp and cc < sc:
        signals.append("Granville_Sell6_FallBack")
    if sma_falling and cp < sp and cp2 < sp2 and cp > cp2 and cc < cp and abs(cp - sp) / sp < 0.02:
        signals.append("Granville_Sell7_NearSMA")
    if sma_rising and deviation > 0.10:
        signals.append("Granville_Sell8_Overbought")

    return signals


# ============================================================
# シグナル検出: SMAアラインメント
# ============================================================

def detect_sma_alignment(df: pd.DataFrame) -> List[str]:
    """SMAパーフェクトオーダーの発生・継続を検出する。"""
    signals: List[str] = []
    if len(df) < 2:
        return signals

    def _check(col_name: str, label: str):
        if col_name in df.columns:
            c = df.iloc[-1][col_name]
            p = df.iloc[-2][col_name]
        else:
            s5, s25, s75, s200 = (f"SMA_{x}" for x in [SMA_SHORT, SMA_MEDIUM, SMA_LONG, SMA_VERY_LONG])
            if not all(x in df.columns for x in [s5, s25, s75, s200]):
                return
            r = df.iloc[-1]
            rp = df.iloc[-2]
            if "Bull" in label:
                c = r[s5] > r[s25] > r[s75] > r[s200]
                p = rp[s5] > rp[s25] > rp[s75] > rp[s200]
            else:
                c = r[s5] < r[s25] < r[s75] < r[s200]
                p = rp[s5] < rp[s25] < rp[s75] < rp[s200]
        if c:
            signals.append(f"SMA_PerfectOrder_{label}_Start" if not p else f"SMA_PerfectOrder_{label}_Continue")

    _check("SMA_Perfect_Order_Bull", "Bull")
    _check("SMA_Perfect_Order_Bear", "Bear")
    return signals
