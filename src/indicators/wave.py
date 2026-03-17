"""エリオット波動（簡易）、フィボナッチ、ダウ理論モジュール"""

import numpy as np
import pandas as pd

from src.config import FIBONACCI_LEVELS


def find_swing_points(
    df: pd.DataFrame,
    window: int = 5,
    col: str = "Close",
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """スイングハイ・スイングローを検出

    Args:
        df: 株価DataFrame
        window: 前後N本で極値判定
        col: 使用するカラム名

    Returns:
        (swing_highs, swing_lows): 各リストは (index, price) のタプル
    """
    prices = df[col].values
    highs = []
    lows = []

    for i in range(window, len(prices) - window):
        # スイングハイ: 前後window本より高い
        if all(prices[i] >= prices[i - j] for j in range(1, window + 1)) and all(
            prices[i] >= prices[i + j] for j in range(1, window + 1)
        ):
            highs.append((i, prices[i]))

        # スイングロー: 前後window本より低い
        if all(prices[i] <= prices[i - j] for j in range(1, window + 1)) and all(
            prices[i] <= prices[i + j] for j in range(1, window + 1)
        ):
            lows.append((i, prices[i]))

    return highs, lows


def calculate_fibonacci_levels(
    high: float,
    low: float,
    direction: str = "up",
) -> dict[str, float]:
    """フィボナッチリトレースメント水準を計算

    Args:
        high: 直近の高値
        low: 直近の安値
        direction: "up"=上昇トレンドの押し目, "down"=下降トレンドの戻り

    Returns:
        各フィボナッチ水準の価格辞書
    """
    diff = high - low
    levels = {}

    if direction == "up":
        # 上昇トレンド: 高値からの押し目水準
        for level in FIBONACCI_LEVELS:
            levels[f"fib_{level:.3f}"] = high - diff * level
    else:
        # 下降トレンド: 安値からの戻り水準
        for level in FIBONACCI_LEVELS:
            levels[f"fib_{level:.3f}"] = low + diff * level

    levels["fib_high"] = high
    levels["fib_low"] = low
    return levels


def detect_fibonacci_support_resistance(
    df: pd.DataFrame,
    lookback: int = 60,
) -> dict:
    """直近の高値・安値からフィボナッチ水準を計算し、現在値との関係を返す

    Returns:
        {
            "levels": {水準名: 価格},
            "nearest_support": 最寄りサポート水準,
            "nearest_resistance": 最寄りレジスタンス水準,
            "position": "above_618" / "between_382_618" / "below_382" etc.
        }
    """
    if len(df) < lookback:
        return {}

    recent = df.tail(lookback)
    high = recent["High"].max()
    low = recent["Low"].min()
    close = df["Close"].iloc[-1]

    # トレンド方向判定
    mid_idx = len(recent) // 2
    first_half_avg = recent["Close"].iloc[:mid_idx].mean()
    second_half_avg = recent["Close"].iloc[mid_idx:].mean()
    direction = "up" if second_half_avg > first_half_avg else "down"

    levels = calculate_fibonacci_levels(high, low, direction)

    # 最寄りのサポート・レジスタンスを探す
    level_prices = sorted(
        [(name, price) for name, price in levels.items() if name.startswith("fib_")],
        key=lambda x: x[1],
    )

    nearest_support = None
    nearest_resistance = None
    for name, price in level_prices:
        if price < close:
            nearest_support = (name, price)
        elif price > close and nearest_resistance is None:
            nearest_resistance = (name, price)

    return {
        "levels": levels,
        "direction": direction,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
    }


def detect_dow_theory_trend(df: pd.DataFrame, window: int = 10) -> str:
    """ダウ理論に基づくトレンド判定

    上昇トレンド: 高値切り上げ＋安値切り上げ
    下降トレンド: 高値切り下げ＋安値切り下げ
    横ばい: それ以外

    Returns:
        "uptrend", "downtrend", or "sideways"
    """
    highs, lows = find_swing_points(df, window=window)

    if len(highs) < 2 or len(lows) < 2:
        return "sideways"

    # 直近2つのスイングポイントを比較
    recent_highs = highs[-2:]
    recent_lows = lows[-2:]

    higher_highs = recent_highs[1][1] > recent_highs[0][1]
    higher_lows = recent_lows[1][1] > recent_lows[0][1]
    lower_highs = recent_highs[1][1] < recent_highs[0][1]
    lower_lows = recent_lows[1][1] < recent_lows[0][1]

    if higher_highs and higher_lows:
        return "uptrend"
    elif lower_highs and lower_lows:
        return "downtrend"
    else:
        return "sideways"


def detect_elliott_wave_position(
    df: pd.DataFrame,
    window: int = 5,
) -> dict:
    """エリオット波動の簡易的な位置推定

    スイングポイントの数とパターンから、現在が推進波(1-5)のどの位置か、
    または修正波(A-B-C)のどの位置かを推定する。

    Returns:
        {
            "wave_count": スイングポイント数,
            "estimated_position": "wave_3_up" / "wave_5_up" / "wave_a_down" etc.,
            "confidence": 0.0-1.0
        }
    """
    highs, lows = find_swing_points(df, window=window)

    if len(highs) < 2 or len(lows) < 2:
        return {"wave_count": 0, "estimated_position": "unknown", "confidence": 0.0}

    # 全スイングポイントを時系列順に並べる
    all_points = [(i, p, "high") for i, p in highs] + [
        (i, p, "low") for i, p in lows
    ]
    all_points.sort(key=lambda x: x[0])

    # 直近5-7個のスイングポイントでパターンを判定
    recent = all_points[-7:]
    wave_count = len(recent)

    # 簡易判定: 上昇推進波のパターン
    if wave_count >= 5:
        prices = [p[1] for p in recent[-5:]]
        types = [p[2] for p in recent[-5:]]

        # 推進波5波: L-H-L-H-L or H-L-H-L-H パターン
        if types == ["low", "high", "low", "high", "low"]:
            if prices[1] > prices[3]:
                return {
                    "wave_count": wave_count,
                    "estimated_position": "wave_c_down",
                    "confidence": 0.4,
                }
        elif types == ["high", "low", "high", "low", "high"]:
            if prices[0] < prices[2] < prices[4]:
                return {
                    "wave_count": wave_count,
                    "estimated_position": "wave_5_up",
                    "confidence": 0.4,
                }
            elif prices[0] < prices[2] and prices[2] > prices[4]:
                return {
                    "wave_count": wave_count,
                    "estimated_position": "wave_a_down",
                    "confidence": 0.3,
                }

    # デフォルト: トレンド方向から推定
    dow_trend = detect_dow_theory_trend(df, window=window)
    if dow_trend == "uptrend":
        position = "wave_3_up"
    elif dow_trend == "downtrend":
        position = "wave_3_down"
    else:
        position = "sideways"

    return {
        "wave_count": wave_count,
        "estimated_position": position,
        "confidence": 0.2,
    }


def get_wave_signals(df: pd.DataFrame) -> list[str]:
    """波動分析シグナルを取得

    Returns:
        シグナル名のリスト
    """
    signals = []

    # ダウ理論トレンド
    dow_trend = detect_dow_theory_trend(df)
    if dow_trend == "uptrend":
        signals.append("ダウ理論:上昇トレンド")
    elif dow_trend == "downtrend":
        signals.append("ダウ理論:下降トレンド")

    # フィボナッチ水準
    fib = detect_fibonacci_support_resistance(df)
    if fib:
        close = df["Close"].iloc[-1]
        if fib.get("nearest_support"):
            name, price = fib["nearest_support"]
            pct = (close - price) / close * 100
            if pct < 2.0:  # サポート付近（2%以内）
                signals.append(f"フィボナッチサポート付近({name})")
        if fib.get("nearest_resistance"):
            name, price = fib["nearest_resistance"]
            pct = (price - close) / close * 100
            if pct < 2.0:  # レジスタンス付近（2%以内）
                signals.append(f"フィボナッチレジスタンス付近({name})")

    # エリオット波動
    elliott = detect_elliott_wave_position(df)
    if elliott["confidence"] >= 0.3:
        pos = elliott["estimated_position"]
        if "wave_3_up" in pos:
            signals.append("エリオット:第3波上昇の可能性")
        elif "wave_5_up" in pos:
            signals.append("エリオット:第5波上昇（天井注意）")
        elif "wave_a_down" in pos:
            signals.append("エリオット:A波下落の可能性")

    return signals
