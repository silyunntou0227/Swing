"""ローソク足・チャートパターン検出モジュール

包み足（Engulfing）、はらみ足（Harami）、十字線（Doji）、
ハンマー / 流れ星、三兵（Three Soldiers / Crows）を検出する。
DataFrameには Date, Open, High, Low, Close, Volume カラムが必要。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ==================================================================
# ヘルパー
# ==================================================================

def _body(row: pd.Series) -> float:
    """実体の大きさ（絶対値）を返す。"""
    return abs(row["Close"] - row["Open"])


def _range(row: pd.Series) -> float:
    """高値 - 安値（ヒゲ含む全体幅）を返す。"""
    return row["High"] - row["Low"]


def _is_bullish(row: pd.Series) -> bool:
    """陽線かどうかを返す。"""
    return row["Close"] > row["Open"]


def _is_bearish(row: pd.Series) -> bool:
    """陰線かどうかを返す。"""
    return row["Close"] < row["Open"]


def _upper_shadow(row: pd.Series) -> float:
    """上ヒゲの長さを返す。"""
    return row["High"] - max(row["Open"], row["Close"])


def _lower_shadow(row: pd.Series) -> float:
    """下ヒゲの長さを返す。"""
    return min(row["Open"], row["Close"]) - row["Low"]


# ==================================================================
# 包み足（Engulfing）
# ==================================================================

def detect_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """包み足（Bullish / Bearish Engulfing）を検出する。

    当日の実体が前日の実体を完全に包む場合に検出。

    追加カラム:
        - Engulfing: ``1`` = 強気包み足, ``-1`` = 弱気包み足, ``0`` = なし

    Args:
        df: OHLCV DataFrame。

    Returns:
        Engulfing カラムが追加された DataFrame。
    """
    df = df.copy()
    signals = np.zeros(len(df), dtype=int)

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        prev_open, prev_close = prev["Open"], prev["Close"]
        curr_open, curr_close = curr["Open"], curr["Close"]

        # 強気包み足: 前日陰線 + 当日陽線で前日実体を包む
        if (prev_close < prev_open
                and curr_close > curr_open
                and curr_open <= prev_close
                and curr_close >= prev_open):
            signals[i] = 1

        # 弱気包み足: 前日陽線 + 当日陰線で前日実体を包む
        elif (prev_close > prev_open
              and curr_close < curr_open
              and curr_open >= prev_close
              and curr_close <= prev_open):
            signals[i] = -1

    df["Engulfing"] = signals
    return df


# ==================================================================
# はらみ足（Harami）
# ==================================================================

def detect_harami(df: pd.DataFrame) -> pd.DataFrame:
    """はらみ足（Bullish / Bearish Harami）を検出する。

    当日の実体が前日の実体の内側に収まる場合に検出。

    追加カラム:
        - Harami: ``1`` = 強気はらみ足, ``-1`` = 弱気はらみ足, ``0`` = なし

    Args:
        df: OHLCV DataFrame。

    Returns:
        Harami カラムが追加された DataFrame。
    """
    df = df.copy()
    signals = np.zeros(len(df), dtype=int)

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        prev_top = max(prev["Open"], prev["Close"])
        prev_bot = min(prev["Open"], prev["Close"])
        curr_top = max(curr["Open"], curr["Close"])
        curr_bot = min(curr["Open"], curr["Close"])

        # 当日実体が前日実体の内側
        if curr_top <= prev_top and curr_bot >= prev_bot and _body(prev) > 0:
            # 強気はらみ: 前日陰線 + 当日陽線
            if _is_bearish(prev) and _is_bullish(curr):
                signals[i] = 1
            # 弱気はらみ: 前日陽線 + 当日陰線
            elif _is_bullish(prev) and _is_bearish(curr):
                signals[i] = -1

    df["Harami"] = signals
    return df


# ==================================================================
# 十字線（Doji）
# ==================================================================

def detect_doji(df: pd.DataFrame, body_ratio: float = 0.10) -> pd.DataFrame:
    """十字線（Doji）を検出する。

    実体が全体レンジの *body_ratio* 未満の場合に十字線と判定。

    追加カラム:
        - Doji: ``True`` = 十字線, ``False`` = なし

    Args:
        df: OHLCV DataFrame。
        body_ratio: 実体 / レンジ の閾値（デフォルト 0.10 = 10%）。

    Returns:
        Doji カラムが追加された DataFrame。
    """
    df = df.copy()
    bodies = (df["Close"] - df["Open"]).abs()
    ranges = df["High"] - df["Low"]
    # レンジがゼロ（始値=高値=安値=終値）の場合も十字線扱い
    df["Doji"] = np.where(ranges == 0, True, bodies / ranges < body_ratio)
    df["Doji"] = df["Doji"].astype(bool)
    return df


# ==================================================================
# ハンマー / 流れ星（Hammer / Shooting Star）
# ==================================================================

def detect_hammer(
    df: pd.DataFrame,
    shadow_ratio: float = 2.0,
    small_shadow_ratio: float = 0.3,
) -> pd.DataFrame:
    """ハンマー（Hammer）と流れ星（Shooting Star）を検出する。

    ハンマー: 下ヒゲが実体の *shadow_ratio* 倍以上、上ヒゲが実体の
              *small_shadow_ratio* 倍以下。下落トレンド中の反転示唆。
    流れ星:   上ヒゲが実体の *shadow_ratio* 倍以上、下ヒゲが実体の
              *small_shadow_ratio* 倍以下。上昇トレンド中の反転示唆。

    追加カラム:
        - Hammer: ``1`` = ハンマー, ``-1`` = 流れ星, ``0`` = なし

    Args:
        df: OHLCV DataFrame。
        shadow_ratio: ヒゲ / 実体 の最低倍率。
        small_shadow_ratio: 反対側ヒゲ / 実体 の最大倍率。

    Returns:
        Hammer カラムが追加された DataFrame。
    """
    df = df.copy()
    signals = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        row = df.iloc[i]
        body = _body(row)
        if body == 0:
            continue

        upper = _upper_shadow(row)
        lower = _lower_shadow(row)

        # ハンマー: 長い下ヒゲ + 短い上ヒゲ
        if lower >= body * shadow_ratio and upper <= body * small_shadow_ratio:
            signals[i] = 1

        # 流れ星: 長い上ヒゲ + 短い下ヒゲ
        elif upper >= body * shadow_ratio and lower <= body * small_shadow_ratio:
            signals[i] = -1

    df["Hammer"] = signals
    return df


# ==================================================================
# 三兵（Three White Soldiers / Three Black Crows）
# ==================================================================

def detect_three_soldiers_crows(df: pd.DataFrame) -> pd.DataFrame:
    """赤三兵（Three White Soldiers）/ 三羽烏（Three Black Crows）を検出する。

    赤三兵: 3連続陽線で、各足の終値が前日終値より高く、
            各足の始値が前日の実体内に収まる。
    三羽烏: 3連続陰線で、各足の終値が前日終値より低く、
            各足の始値が前日の実体内に収まる。

    追加カラム:
        - ThreeSoldiersCrows: ``1`` = 赤三兵, ``-1`` = 三羽烏, ``0`` = なし

    Args:
        df: OHLCV DataFrame。

    Returns:
        ThreeSoldiersCrows カラムが追加された DataFrame。
    """
    df = df.copy()
    signals = np.zeros(len(df), dtype=int)

    for i in range(2, len(df)):
        r0 = df.iloc[i - 2]
        r1 = df.iloc[i - 1]
        r2 = df.iloc[i]

        # 赤三兵チェック
        if (_is_bullish(r0) and _is_bullish(r1) and _is_bullish(r2)
                and r1["Close"] > r0["Close"]
                and r2["Close"] > r1["Close"]
                and r0["Open"] <= r1["Open"] <= r0["Close"]
                and r1["Open"] <= r2["Open"] <= r1["Close"]):
            signals[i] = 1

        # 三羽烏チェック
        elif (_is_bearish(r0) and _is_bearish(r1) and _is_bearish(r2)
              and r1["Close"] < r0["Close"]
              and r2["Close"] < r1["Close"]
              and r0["Close"] <= r1["Open"] <= r0["Open"]
              and r1["Close"] <= r2["Open"] <= r1["Open"]):
            signals[i] = -1

    df["ThreeSoldiersCrows"] = signals
    return df


# ==================================================================
# 全パターンシグナル取得
# ==================================================================

def get_pattern_signals(df: pd.DataFrame) -> list[str]:
    """直近（最新）バーで検出された全ローソク足パターンを文字列リストで返す。

    内部で全パターン検出関数を実行し、最新バーの結果をまとめる。

    Args:
        df: OHLCV DataFrame。

    Returns:
        検出されたパターン文字列のリスト。
        例: ["強気包み足 (Bullish Engulfing)", "十字線 (Doji)"]
    """
    df = detect_engulfing(df)
    df = detect_harami(df)
    df = detect_doji(df)
    df = detect_hammer(df)
    df = detect_three_soldiers_crows(df)

    if df.empty:
        return []

    signals: list[str] = []
    last = df.iloc[-1]

    # 包み足
    engulfing = last.get("Engulfing", 0)
    if engulfing == 1:
        signals.append("強気包み足 (Bullish Engulfing)")
    elif engulfing == -1:
        signals.append("弱気包み足 (Bearish Engulfing)")

    # はらみ足
    harami = last.get("Harami", 0)
    if harami == 1:
        signals.append("強気はらみ足 (Bullish Harami)")
    elif harami == -1:
        signals.append("弱気はらみ足 (Bearish Harami)")

    # 十字線
    if last.get("Doji", False):
        signals.append("十字線 (Doji)")

    # ハンマー / 流れ星
    hammer = last.get("Hammer", 0)
    if hammer == 1:
        signals.append("ハンマー (Hammer)")
    elif hammer == -1:
        signals.append("流れ星 (Shooting Star)")

    # 三兵
    tsc = last.get("ThreeSoldiersCrows", 0)
    if tsc == 1:
        signals.append("赤三兵 (Three White Soldiers)")
    elif tsc == -1:
        signals.append("三羽烏 (Three Black Crows)")

    return signals
