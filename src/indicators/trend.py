"""トレンド系テクニカル指標モジュール

移動平均線、MACD、一目均衡表、ADX などのトレンド系指標の算出と
ゴールデンクロス／デッドクロス、グランビルの法則等のシグナル検出を行う。

対象DataFrame列: Date, Open, High, Low, Close, Volume
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
import pandas_ta as ta

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
    """SMA (5, 25, 75, 200) を算出して返す。

    Parameters
    ----------
    df : pd.DataFrame
        ``Close`` 列を含む株価データ。

    Returns
    -------
    pd.DataFrame
        各期間のSMA列を持つDataFrame。
    """
    result = pd.DataFrame(index=df.index)
    for period, label in [
        (SMA_SHORT, f"SMA_{SMA_SHORT}"),
        (SMA_MEDIUM, f"SMA_{SMA_MEDIUM}"),
        (SMA_LONG, f"SMA_{SMA_LONG}"),
        (SMA_VERY_LONG, f"SMA_{SMA_VERY_LONG}"),
    ]:
        result[label] = ta.sma(df["Close"], length=period)
    return result


def calc_ema(df: pd.DataFrame) -> pd.DataFrame:
    """EMA (12, 26) を算出して返す。

    Parameters
    ----------
    df : pd.DataFrame
        ``Close`` 列を含む株価データ。

    Returns
    -------
    pd.DataFrame
        各期間のEMA列を持つDataFrame。
    """
    result = pd.DataFrame(index=df.index)
    for period, label in [
        (EMA_SHORT, f"EMA_{EMA_SHORT}"),
        (EMA_LONG, f"EMA_{EMA_LONG}"),
    ]:
        result[label] = ta.ema(df["Close"], length=period)
    return result


# ============================================================
# MACD
# ============================================================


def calc_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD (12, 26, 9) を算出して返す。

    算出列:
        - MACD: MACDライン (EMA12 - EMA26)
        - MACD_Signal: シグナルライン (MACDの9期間EMA)
        - MACD_Hist: ヒストグラム (MACD - Signal)

    Parameters
    ----------
    df : pd.DataFrame
        ``Close`` 列を含む株価データ。

    Returns
    -------
    pd.DataFrame
        MACD / Signal / Histogram の3列。
    """
    macd_df = ta.macd(
        df["Close"],
        fast=EMA_SHORT,
        slow=EMA_LONG,
        signal=MACD_SIGNAL,
    )
    # pandas_ta の列名を統一名称にリネーム
    macd_df.columns = ["MACD", "MACD_Hist", "MACD_Signal"]
    return macd_df


# ============================================================
# 一目均衡表
# ============================================================


def calc_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """一目均衡表の各線を算出して返す。

    算出列:
        - Ichimoku_Tenkan: 転換線 (9期間)
        - Ichimoku_Kijun: 基準線 (26期間)
        - Ichimoku_SenkouA: 先行スパンA
        - Ichimoku_SenkouB: 先行スパンB (52期間)
        - Ichimoku_Chikou: 遅行スパン

    Parameters
    ----------
    df : pd.DataFrame
        ``High``, ``Low``, ``Close`` 列を含む株価データ。

    Returns
    -------
    pd.DataFrame
        一目均衡表の5本の線。
    """
    ichimoku_df, _ = ta.ichimoku(
        df["High"],
        df["Low"],
        df["Close"],
        tenkan=ICHIMOKU_TENKAN,
        kijun=ICHIMOKU_KIJUN,
        senkou=ICHIMOKU_SENKOU_B,
    )

    # pandas_ta の列名を統一名称にリネーム
    rename_map = {}
    for col in ichimoku_df.columns:
        col_lower = col.lower()
        if "tenkan" in col_lower or "isa" in col_lower.replace("_", ""):
            if "tenkan" in col_lower:
                rename_map[col] = "Ichimoku_Tenkan"
        if "kijun" in col_lower or "isb" in col_lower.replace("_", ""):
            if "kijun" in col_lower:
                rename_map[col] = "Ichimoku_Kijun"
        if "isa" in col_lower:
            rename_map[col] = "Ichimoku_SenkouA"
        if "isb" in col_lower:
            rename_map[col] = "Ichimoku_SenkouB"
        if "ics" in col_lower:
            rename_map[col] = "Ichimoku_Chikou"

    # pandas_ta の ichimoku は ISA/ISB/ITS/IKS/ICS という列名を返す
    # 万が一マッピングが不完全な場合はデフォルト列名で対応
    if len(rename_map) < len(ichimoku_df.columns):
        expected = [
            "Ichimoku_Tenkan",
            "Ichimoku_Kijun",
            "Ichimoku_SenkouA",
            "Ichimoku_SenkouB",
            "Ichimoku_Chikou",
        ]
        cols = list(ichimoku_df.columns)
        for i, col in enumerate(cols):
            if col not in rename_map and i < len(expected):
                rename_map[col] = expected[i]

    ichimoku_df = ichimoku_df.rename(columns=rename_map)

    # 先行スパンの列だけが返る場合、遅行スパンを手動計算
    if "Ichimoku_Chikou" not in ichimoku_df.columns:
        ichimoku_df["Ichimoku_Chikou"] = df["Close"].shift(-ICHIMOKU_KIJUN)

    # 転換線・基準線が欠落している場合も手動計算
    if "Ichimoku_Tenkan" not in ichimoku_df.columns:
        high_t = df["High"].rolling(window=ICHIMOKU_TENKAN).max()
        low_t = df["Low"].rolling(window=ICHIMOKU_TENKAN).min()
        ichimoku_df["Ichimoku_Tenkan"] = (high_t + low_t) / 2

    if "Ichimoku_Kijun" not in ichimoku_df.columns:
        high_k = df["High"].rolling(window=ICHIMOKU_KIJUN).max()
        low_k = df["Low"].rolling(window=ICHIMOKU_KIJUN).min()
        ichimoku_df["Ichimoku_Kijun"] = (high_k + low_k) / 2

    return ichimoku_df


# ============================================================
# ADX (Average Directional Index)
# ============================================================


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX, +DI, -DI を算出して返す。

    Parameters
    ----------
    df : pd.DataFrame
        ``High``, ``Low``, ``Close`` 列を含む株価データ。
    period : int, optional
        ADXの計算期間 (デフォルト: 14)。

    Returns
    -------
    pd.DataFrame
        ADX / +DI / -DI の3列。
    """
    adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=period)
    # pandas_ta は ADX_14, DMP_14, DMN_14 のような列名を返す
    rename_map = {}
    for col in adx_df.columns:
        col_upper = col.upper()
        if col_upper.startswith("ADX"):
            rename_map[col] = "ADX"
        elif col_upper.startswith("DMP"):
            rename_map[col] = "ADX_Plus_DI"
        elif col_upper.startswith("DMN"):
            rename_map[col] = "ADX_Minus_DI"
    adx_df = adx_df.rename(columns=rename_map)
    return adx_df


# ============================================================
# 全トレンド指標一括算出
# ============================================================


def calculate_all_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """全トレンド系テクニカル指標をDataFrameに追加して返す。

    元のDataFrameは変更せず、指標列を追加したコピーを返す。

    Parameters
    ----------
    df : pd.DataFrame
        Date, Open, High, Low, Close, Volume 列を含む株価データ。

    Returns
    -------
    pd.DataFrame
        全トレンド指標列が追加されたDataFrame。
    """
    result = df.copy()

    # SMA
    sma_df = calc_sma(result)
    for col in sma_df.columns:
        result[col] = sma_df[col]

    # EMA
    ema_df = calc_ema(result)
    for col in ema_df.columns:
        result[col] = ema_df[col]

    # MACD
    macd_df = calc_macd(result)
    for col in macd_df.columns:
        result[col] = macd_df[col]

    # 一目均衡表
    ichimoku_df = calc_ichimoku(result)
    for col in ichimoku_df.columns:
        result[col] = ichimoku_df[col]

    # ADX
    adx_df = calc_adx(result)
    for col in adx_df.columns:
        result[col] = adx_df[col]

    # SMAアラインメント
    result["SMA_Perfect_Order_Bull"] = _is_perfect_order_bull(result)
    result["SMA_Perfect_Order_Bear"] = _is_perfect_order_bear(result)

    logger.debug(
        "トレンド指標算出完了: %d 行 x %d 列",
        len(result),
        len(result.columns) - len(df.columns),
    )
    return result


# ============================================================
# SMAアラインメント (パーフェクトオーダー)
# ============================================================


def _is_perfect_order_bull(df: pd.DataFrame) -> pd.Series:
    """上昇パーフェクトオーダー判定: SMA5 > SMA25 > SMA75 > SMA200"""
    s5 = f"SMA_{SMA_SHORT}"
    s25 = f"SMA_{SMA_MEDIUM}"
    s75 = f"SMA_{SMA_LONG}"
    s200 = f"SMA_{SMA_VERY_LONG}"
    return (
        (df[s5] > df[s25])
        & (df[s25] > df[s75])
        & (df[s75] > df[s200])
    )


def _is_perfect_order_bear(df: pd.DataFrame) -> pd.Series:
    """下降パーフェクトオーダー判定: SMA5 < SMA25 < SMA75 < SMA200"""
    s5 = f"SMA_{SMA_SHORT}"
    s25 = f"SMA_{SMA_MEDIUM}"
    s75 = f"SMA_{SMA_LONG}"
    s200 = f"SMA_{SMA_VERY_LONG}"
    return (
        (df[s5] < df[s25])
        & (df[s25] < df[s75])
        & (df[s75] < df[s200])
    )


# ============================================================
# シグナル検出: ゴールデンクロス / デッドクロス
# ============================================================


def detect_golden_dead_cross(df: pd.DataFrame) -> List[str]:
    """SMA5 と SMA25 のゴールデンクロス / デッドクロスを検出する。

    直近1本のバーで発生したクロスのみ返す。

    Parameters
    ----------
    df : pd.DataFrame
        ``SMA_{SMA_SHORT}``, ``SMA_{SMA_MEDIUM}`` 列を含むDataFrame。

    Returns
    -------
    List[str]
        検出されたシグナル名のリスト。
    """
    signals: List[str] = []
    s5 = f"SMA_{SMA_SHORT}"
    s25 = f"SMA_{SMA_MEDIUM}"

    if len(df) < 2:
        return signals

    if s5 not in df.columns or s25 not in df.columns:
        logger.warning("SMA列が見つかりません: %s, %s", s5, s25)
        return signals

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # ゴールデンクロス: 前日 SMA5 <= SMA25 → 当日 SMA5 > SMA25
    if prev[s5] <= prev[s25] and curr[s5] > curr[s25]:
        signals.append("SMA_GoldenCross_5_25")

    # デッドクロス: 前日 SMA5 >= SMA25 → 当日 SMA5 < SMA25
    if prev[s5] >= prev[s25] and curr[s5] < curr[s25]:
        signals.append("SMA_DeadCross_5_25")

    return signals


# ============================================================
# シグナル検出: MACDクロス
# ============================================================


def detect_macd_cross(df: pd.DataFrame) -> List[str]:
    """MACDラインとシグナルラインのクロスを検出する。

    直近1本のバーで発生したクロスのみ返す。
    ゼロライン上下も判定に含める。

    Parameters
    ----------
    df : pd.DataFrame
        ``MACD``, ``MACD_Signal`` 列を含むDataFrame。

    Returns
    -------
    List[str]
        検出されたシグナル名のリスト。
    """
    signals: List[str] = []

    if len(df) < 2:
        return signals

    for col in ("MACD", "MACD_Signal"):
        if col not in df.columns:
            logger.warning("MACD列が見つかりません: %s", col)
            return signals

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # MACDがシグナルを上抜け（買いシグナル）
    if prev["MACD"] <= prev["MACD_Signal"] and curr["MACD"] > curr["MACD_Signal"]:
        if curr["MACD"] < 0:
            signals.append("MACD_BullishCross_BelowZero")
        else:
            signals.append("MACD_BullishCross_AboveZero")

    # MACDがシグナルを下抜け（売りシグナル）
    if prev["MACD"] >= prev["MACD_Signal"] and curr["MACD"] < curr["MACD_Signal"]:
        if curr["MACD"] > 0:
            signals.append("MACD_BearishCross_AboveZero")
        else:
            signals.append("MACD_BearishCross_BelowZero")

    # ゼロラインクロス
    if prev["MACD"] <= 0 and curr["MACD"] > 0:
        signals.append("MACD_ZeroCross_Bullish")
    if prev["MACD"] >= 0 and curr["MACD"] < 0:
        signals.append("MACD_ZeroCross_Bearish")

    return signals


# ============================================================
# シグナル検出: 一目均衡表
# ============================================================


def detect_ichimoku_signals(df: pd.DataFrame) -> List[str]:
    """一目均衡表のシグナルを検出する。

    検出項目:
        - 雲ブレイクアウト / ブレイクダウン（終値と先行スパンの関係）
        - 転換線 / 基準線クロス
        - 三役好転 / 三役逆転

    Parameters
    ----------
    df : pd.DataFrame
        一目均衡表の各列を含むDataFrame。

    Returns
    -------
    List[str]
        検出されたシグナル名のリスト。
    """
    signals: List[str] = []

    required = [
        "Ichimoku_Tenkan",
        "Ichimoku_Kijun",
        "Ichimoku_SenkouA",
        "Ichimoku_SenkouB",
    ]
    for col in required:
        if col not in df.columns:
            logger.warning("一目均衡表列が見つかりません: %s", col)
            return signals

    if len(df) < 2:
        return signals

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    cloud_top = max(curr["Ichimoku_SenkouA"], curr["Ichimoku_SenkouB"])
    cloud_bottom = min(curr["Ichimoku_SenkouA"], curr["Ichimoku_SenkouB"])
    prev_cloud_top = max(prev["Ichimoku_SenkouA"], prev["Ichimoku_SenkouB"])
    prev_cloud_bottom = min(prev["Ichimoku_SenkouA"], prev["Ichimoku_SenkouB"])

    # --- 雲ブレイクアウト / ブレイクダウン ---
    if prev["Close"] <= prev_cloud_top and curr["Close"] > cloud_top:
        signals.append("Ichimoku_CloudBreakout_Bullish")
    if prev["Close"] >= prev_cloud_bottom and curr["Close"] < cloud_bottom:
        signals.append("Ichimoku_CloudBreakout_Bearish")

    # --- 転換線 / 基準線クロス ---
    if (
        prev["Ichimoku_Tenkan"] <= prev["Ichimoku_Kijun"]
        and curr["Ichimoku_Tenkan"] > curr["Ichimoku_Kijun"]
    ):
        signals.append("Ichimoku_TenkanKijun_BullishCross")
    if (
        prev["Ichimoku_Tenkan"] >= prev["Ichimoku_Kijun"]
        and curr["Ichimoku_Tenkan"] < curr["Ichimoku_Kijun"]
    ):
        signals.append("Ichimoku_TenkanKijun_BearishCross")

    # --- 三役好転（買い）: 転換線>基準線, 終値>雲, 遅行スパン>過去の終値 ---
    is_tenkan_above = curr["Ichimoku_Tenkan"] > curr["Ichimoku_Kijun"]
    is_above_cloud = curr["Close"] > cloud_top

    chikou_bullish = False
    if "Ichimoku_Chikou" in df.columns and len(df) > ICHIMOKU_KIJUN:
        chikou_idx = len(df) - 1 - ICHIMOKU_KIJUN
        if chikou_idx >= 0:
            chikou_val = df.iloc[-1]["Ichimoku_Chikou"]
            past_close = df.iloc[chikou_idx]["Close"]
            if pd.notna(chikou_val) and pd.notna(past_close):
                chikou_bullish = chikou_val > past_close

    if is_tenkan_above and is_above_cloud and chikou_bullish:
        signals.append("Ichimoku_Sannyaku_Kouten")  # 三役好転

    # --- 三役逆転（売り）: 転換線<基準線, 終値<雲, 遅行スパン<過去の終値 ---
    is_tenkan_below = curr["Ichimoku_Tenkan"] < curr["Ichimoku_Kijun"]
    is_below_cloud = curr["Close"] < cloud_bottom

    chikou_bearish = False
    if "Ichimoku_Chikou" in df.columns and len(df) > ICHIMOKU_KIJUN:
        chikou_idx = len(df) - 1 - ICHIMOKU_KIJUN
        if chikou_idx >= 0:
            chikou_val = df.iloc[-1]["Ichimoku_Chikou"]
            past_close = df.iloc[chikou_idx]["Close"]
            if pd.notna(chikou_val) and pd.notna(past_close):
                chikou_bearish = chikou_val < past_close

    if is_tenkan_below and is_below_cloud and chikou_bearish:
        signals.append("Ichimoku_Sannyaku_Gyakuten")  # 三役逆転

    return signals


# ============================================================
# シグナル検出: グランビルの法則
# ============================================================


def detect_granville_signals(
    df: pd.DataFrame,
    sma_col: str = f"SMA_{SMA_MEDIUM}",
    close_col: str = "Close",
) -> List[str]:
    """グランビルの法則（8つのルール）に基づくシグナルを検出する。

    買いシグナル (4つ):
        1. SMA上昇中にCloseがSMAを下から上抜け
        2. SMA上昇中にCloseがSMA割れ後に反発
        3. SMA上昇中にCloseがSMA付近で反発（SMA割れず）
        4. SMA下降中にCloseがSMAから大幅乖離（売られ過ぎ反発）

    売りシグナル (4つ):
        5. SMA下降中にCloseがSMAを上から下抜け
        6. SMA下降中にCloseがSMA超え後に反落
        7. SMA下降中にCloseがSMA付近で反落（SMA超えず）
        8. SMA上昇中にCloseがSMAから大幅乖離（買われ過ぎ反落）

    Parameters
    ----------
    df : pd.DataFrame
        SMA列とClose列を含むDataFrame。
    sma_col : str
        使用するSMA列名。
    close_col : str
        使用する終値列名。

    Returns
    -------
    List[str]
        検出されたシグナル名のリスト。
    """
    signals: List[str] = []

    if sma_col not in df.columns:
        logger.warning("SMA列が見つかりません: %s", sma_col)
        return signals

    if len(df) < 4:
        return signals

    # 直近4本分のデータ
    d = df[[close_col, sma_col]].iloc[-4:].copy()
    d["SMA_Diff"] = d[sma_col].diff()  # SMAの傾き

    curr = d.iloc[-1]
    prev = d.iloc[-2]
    prev2 = d.iloc[-3]

    sma_rising = curr["SMA_Diff"] > 0
    sma_falling = curr["SMA_Diff"] < 0

    close_curr = curr[close_col]
    close_prev = prev[close_col]
    close_prev2 = prev2[close_col]
    sma_curr = curr[sma_col]
    sma_prev = prev[sma_col]
    sma_prev2 = prev2[sma_col]

    # 乖離率
    deviation = (close_curr - sma_curr) / sma_curr if sma_curr != 0 else 0

    # --- 買いシグナル ---

    # 買い1: SMA上昇中、下から上抜け
    if sma_rising and close_prev <= sma_prev and close_curr > sma_curr:
        signals.append("Granville_Buy1_CrossAbove")

    # 買い2: SMA上昇中、一時SMA割れ後に反発
    if (
        sma_rising
        and close_prev2 > sma_prev2
        and close_prev < sma_prev
        and close_curr > sma_curr
    ):
        signals.append("Granville_Buy2_BounceBack")

    # 買い3: SMA上昇中、SMA付近で反発（SMA割れず）
    if (
        sma_rising
        and close_prev > sma_prev
        and close_prev2 > sma_prev2
        and close_prev < close_prev2  # 下落中
        and close_curr > close_prev  # 反発
        and abs(close_prev - sma_prev) / sma_prev < 0.02  # SMA付近
    ):
        signals.append("Granville_Buy3_NearSMA")

    # 買い4: SMA下降中、大幅マイナス乖離（売られ過ぎ）
    if sma_falling and deviation < -0.10:
        signals.append("Granville_Buy4_Oversold")

    # --- 売りシグナル ---

    # 売り5: SMA下降中、上から下抜け
    if sma_falling and close_prev >= sma_prev and close_curr < sma_curr:
        signals.append("Granville_Sell5_CrossBelow")

    # 売り6: SMA下降中、一時SMA超え後に反落
    if (
        sma_falling
        and close_prev2 < sma_prev2
        and close_prev > sma_prev
        and close_curr < sma_curr
    ):
        signals.append("Granville_Sell6_FallBack")

    # 売り7: SMA下降中、SMA付近で反落（SMA超えず）
    if (
        sma_falling
        and close_prev < sma_prev
        and close_prev2 < sma_prev2
        and close_prev > close_prev2  # 上昇中
        and close_curr < close_prev  # 反落
        and abs(close_prev - sma_prev) / sma_prev < 0.02  # SMA付近
    ):
        signals.append("Granville_Sell7_NearSMA")

    # 売り8: SMA上昇中、大幅プラス乖離（買われ過ぎ）
    if sma_rising and deviation > 0.10:
        signals.append("Granville_Sell8_Overbought")

    return signals


# ============================================================
# シグナル検出: SMAアラインメント (パーフェクトオーダー)
# ============================================================


def detect_sma_alignment(df: pd.DataFrame) -> List[str]:
    """SMAパーフェクトオーダーの発生・継続を検出する。

    Parameters
    ----------
    df : pd.DataFrame
        SMA_5, SMA_25, SMA_75, SMA_200 列を含むDataFrame。
        ``SMA_Perfect_Order_Bull`` / ``SMA_Perfect_Order_Bear`` があれば直接利用、
        なければ内部で判定する。

    Returns
    -------
    List[str]
        検出されたシグナル名のリスト。
    """
    signals: List[str] = []

    if len(df) < 2:
        return signals

    # パーフェクトオーダー列がなければ必要なSMA列から判定
    if "SMA_Perfect_Order_Bull" in df.columns:
        bull_curr = df.iloc[-1]["SMA_Perfect_Order_Bull"]
        bull_prev = df.iloc[-2]["SMA_Perfect_Order_Bull"]
    else:
        s5 = f"SMA_{SMA_SHORT}"
        s25 = f"SMA_{SMA_MEDIUM}"
        s75 = f"SMA_{SMA_LONG}"
        s200 = f"SMA_{SMA_VERY_LONG}"
        for col in (s5, s25, s75, s200):
            if col not in df.columns:
                logger.warning("SMA列が見つかりません: %s", col)
                return signals
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        bull_curr = curr[s5] > curr[s25] > curr[s75] > curr[s200]
        bull_prev = prev[s5] > prev[s25] > prev[s75] > prev[s200]

    if "SMA_Perfect_Order_Bear" in df.columns:
        bear_curr = df.iloc[-1]["SMA_Perfect_Order_Bear"]
        bear_prev = df.iloc[-2]["SMA_Perfect_Order_Bear"]
    else:
        s5 = f"SMA_{SMA_SHORT}"
        s25 = f"SMA_{SMA_MEDIUM}"
        s75 = f"SMA_{SMA_LONG}"
        s200 = f"SMA_{SMA_VERY_LONG}"
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        bear_curr = curr[s5] < curr[s25] < curr[s75] < curr[s200]
        bear_prev = prev[s5] < prev[s25] < prev[s75] < prev[s200]

    # 上昇パーフェクトオーダー
    if bull_curr:
        if not bull_prev:
            signals.append("SMA_PerfectOrder_Bull_Start")
        else:
            signals.append("SMA_PerfectOrder_Bull_Continue")

    # 下降パーフェクトオーダー
    if bear_curr:
        if not bear_prev:
            signals.append("SMA_PerfectOrder_Bear_Start")
        else:
            signals.append("SMA_PerfectOrder_Bear_Continue")

    return signals
