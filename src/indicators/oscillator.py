"""オシレーター系テクニカル指標モジュール

RSI、ストキャスティクスの算出およびシグナル検出を行う。
ダイバージェンス（逆行現象）の検出にも対応。
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.config import (
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    SIGNAL_LOOKBACK_DAYS,
    STOCH_D,
    STOCH_K,
    STOCH_SMOOTH,
)


# ============================================================
# 指標算出
# ============================================================


def calculate_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """RSI (Relative Strength Index) を算出して列に追加する。

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV データ。``Close`` 列が必要。

    Returns
    -------
    pd.DataFrame
        ``RSI`` 列が追加された DataFrame。
    """
    df = df.copy()
    df["RSI"] = ta.rsi(df["Close"], length=RSI_PERIOD)
    return df


def calculate_stochastic(df: pd.DataFrame) -> pd.DataFrame:
    """ストキャスティクス (%K, %D) を算出して列に追加する。

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV データ。``High``, ``Low``, ``Close`` 列が必要。

    Returns
    -------
    pd.DataFrame
        ``STOCHk``, ``STOCHd`` 列が追加された DataFrame。
    """
    df = df.copy()
    stoch = ta.stoch(
        df["High"],
        df["Low"],
        df["Close"],
        k=STOCH_K,
        d=STOCH_D,
        smooth_k=STOCH_SMOOTH,
    )
    df["STOCHk"] = stoch.iloc[:, 0]
    df["STOCHd"] = stoch.iloc[:, 1]
    return df


# ============================================================
# シグナル検出
# ============================================================


def detect_rsi_signals(df: pd.DataFrame) -> pd.DataFrame:
    """RSI の売られすぎ / 買われすぎ反転シグナルを検出する。

    直近 ``SIGNAL_LOOKBACK_DAYS`` (デフォルト 3) 日以内に
    RSI が 30 を上抜け (bullish) または 70 を下抜け (bearish)
    した場合にフラグを立てる。

    Parameters
    ----------
    df : pd.DataFrame
        ``RSI`` 列が必要。

    Returns
    -------
    pd.DataFrame
        ``RSI_oversold_cross`` (bool) / ``RSI_overbought_cross`` (bool)
        列が追加された DataFrame。
    """
    df = df.copy()
    if "RSI" not in df.columns:
        df = calculate_rsi(df)

    rsi = df["RSI"]
    prev_rsi = rsi.shift(1)

    # RSI が oversold ゾーンから上抜け
    oversold_cross = (prev_rsi < RSI_OVERSOLD) & (rsi >= RSI_OVERSOLD)
    # RSI が overbought ゾーンから下抜け
    overbought_cross = (prev_rsi > RSI_OVERBOUGHT) & (rsi <= RSI_OVERBOUGHT)

    # 直近 N 日以内にシグナルが出たか
    lookback = SIGNAL_LOOKBACK_DAYS
    df["RSI_oversold_cross"] = (
        oversold_cross.rolling(window=lookback, min_periods=1).max().astype(bool)
    )
    df["RSI_overbought_cross"] = (
        overbought_cross.rolling(window=lookback, min_periods=1).max().astype(bool)
    )
    return df


def detect_rsi_divergence(
    df: pd.DataFrame, lookback: int = 20
) -> pd.DataFrame:
    """RSI ダイバージェンスを検出する。

    * **強気ダイバージェンス**: 価格が直近安値を更新しているが RSI は
      安値を切り上げている。
    * **弱気ダイバージェンス**: 価格が直近高値を更新しているが RSI は
      高値を切り下げている。

    Parameters
    ----------
    df : pd.DataFrame
        ``Close`` および ``RSI`` 列が必要。
    lookback : int, default 20
        ダイバージェンス検出に使用するルックバック期間。

    Returns
    -------
    pd.DataFrame
        ``RSI_bull_divergence`` (bool) / ``RSI_bear_divergence`` (bool)
        列が追加された DataFrame。
    """
    df = df.copy()
    if "RSI" not in df.columns:
        df = calculate_rsi(df)

    bull_div = pd.Series(False, index=df.index)
    bear_div = pd.Series(False, index=df.index)

    close = df["Close"].values
    rsi = df["RSI"].values

    for i in range(lookback, len(df)):
        window_close = close[i - lookback : i]
        window_rsi = rsi[i - lookback : i]

        # NaN が含まれる場合はスキップ
        if np.isnan(window_rsi).any() or np.isnan(rsi[i]):
            continue

        # 強気ダイバージェンス: 価格が期間安値を更新 & RSI は安値を切り上げ
        prev_low_idx = int(np.nanargmin(window_close))
        if close[i] < window_close[prev_low_idx]:
            if rsi[i] > window_rsi[prev_low_idx]:
                bull_div.iloc[i] = True

        # 弱気ダイバージェンス: 価格が期間高値を更新 & RSI は高値を切り下げ
        prev_high_idx = int(np.nanargmax(window_close))
        if close[i] > window_close[prev_high_idx]:
            if rsi[i] < window_rsi[prev_high_idx]:
                bear_div.iloc[i] = True

    df["RSI_bull_divergence"] = bull_div
    df["RSI_bear_divergence"] = bear_div
    return df


def detect_stochastic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """ストキャスティクスのクロスオーバーシグナルを検出する。

    * **強気シグナル**: 売られすぎゾーン (< 20) で %K が %D を上抜け。
    * **弱気シグナル**: 買われすぎゾーン (> 80) で %K が %D を下抜け。

    Parameters
    ----------
    df : pd.DataFrame
        ``STOCHk``, ``STOCHd`` 列が必要。

    Returns
    -------
    pd.DataFrame
        ``STOCH_bull_cross`` (bool) / ``STOCH_bear_cross`` (bool)
        列が追加された DataFrame。
    """
    df = df.copy()
    if "STOCHk" not in df.columns or "STOCHd" not in df.columns:
        df = calculate_stochastic(df)

    k = df["STOCHk"]
    d = df["STOCHd"]
    prev_k = k.shift(1)
    prev_d = d.shift(1)

    # 売られすぎゾーンでゴールデンクロス
    bull_cross = (prev_k < prev_d) & (k >= d) & (k < 20)
    # 買われすぎゾーンでデッドクロス
    bear_cross = (prev_k > prev_d) & (k <= d) & (k > 80)

    lookback = SIGNAL_LOOKBACK_DAYS
    df["STOCH_bull_cross"] = (
        bull_cross.rolling(window=lookback, min_periods=1).max().astype(bool)
    )
    df["STOCH_bear_cross"] = (
        bear_cross.rolling(window=lookback, min_periods=1).max().astype(bool)
    )
    return df


def _detect_stochastic_divergence(
    df: pd.DataFrame, lookback: int = 20
) -> pd.DataFrame:
    """ストキャスティクス %K のダイバージェンスを検出する。

    Parameters
    ----------
    df : pd.DataFrame
        ``Close`` および ``STOCHk`` 列が必要。
    lookback : int, default 20
        ダイバージェンス検出に使用するルックバック期間。

    Returns
    -------
    pd.DataFrame
        ``STOCH_bull_divergence`` (bool) / ``STOCH_bear_divergence`` (bool)
        列が追加された DataFrame。
    """
    df = df.copy()
    if "STOCHk" not in df.columns:
        df = calculate_stochastic(df)

    bull_div = pd.Series(False, index=df.index)
    bear_div = pd.Series(False, index=df.index)

    close = df["Close"].values
    stoch_k = df["STOCHk"].values

    for i in range(lookback, len(df)):
        window_close = close[i - lookback : i]
        window_stoch = stoch_k[i - lookback : i]

        if np.isnan(window_stoch).any() or np.isnan(stoch_k[i]):
            continue

        # 強気ダイバージェンス
        prev_low_idx = int(np.nanargmin(window_close))
        if close[i] < window_close[prev_low_idx]:
            if stoch_k[i] > window_stoch[prev_low_idx]:
                bull_div.iloc[i] = True

        # 弱気ダイバージェンス
        prev_high_idx = int(np.nanargmax(window_close))
        if close[i] > window_close[prev_high_idx]:
            if stoch_k[i] < window_stoch[prev_high_idx]:
                bear_div.iloc[i] = True

    df["STOCH_bull_divergence"] = bull_div
    df["STOCH_bear_divergence"] = bear_div
    return df


# ============================================================
# 統合関数
# ============================================================


def calculate_all_oscillators(df: pd.DataFrame) -> pd.DataFrame:
    """全オシレーター指標を一括で算出して列に追加する。

    追加される列:
        RSI, STOCHk, STOCHd,
        RSI_oversold_cross, RSI_overbought_cross,
        RSI_bull_divergence, RSI_bear_divergence,
        STOCH_bull_cross, STOCH_bear_cross,
        STOCH_bull_divergence, STOCH_bear_divergence

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV データ。

    Returns
    -------
    pd.DataFrame
        全オシレーター列が追加された DataFrame。
    """
    df = calculate_rsi(df)
    df = calculate_stochastic(df)
    df = detect_rsi_signals(df)
    df = detect_rsi_divergence(df)
    df = detect_stochastic_signals(df)
    df = _detect_stochastic_divergence(df)
    return df


def get_oscillator_signals(df: pd.DataFrame) -> List[str]:
    """最新バーに発生しているオシレーターシグナルの名前一覧を返す。

    Parameters
    ----------
    df : pd.DataFrame
        ``calculate_all_oscillators`` 適用済みの DataFrame。
        未適用の場合は内部で自動算出する。

    Returns
    -------
    List[str]
        検出されたシグナル名のリスト。例:
        ``["RSI_oversold_cross", "STOCH_bull_divergence"]``
    """
    signal_columns = [
        "RSI_oversold_cross",
        "RSI_overbought_cross",
        "RSI_bull_divergence",
        "RSI_bear_divergence",
        "STOCH_bull_cross",
        "STOCH_bear_cross",
        "STOCH_bull_divergence",
        "STOCH_bear_divergence",
    ]

    # 必要な列が揃っていなければ全指標を算出
    if not all(col in df.columns for col in signal_columns):
        df = calculate_all_oscillators(df)

    if df.empty:
        return []

    last = df.iloc[-1]
    signals: List[str] = []
    for col in signal_columns:
        if pd.notna(last.get(col)) and bool(last[col]):
            signals.append(col)

    return signals
