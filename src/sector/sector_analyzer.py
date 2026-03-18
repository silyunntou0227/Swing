"""セクターローテーション分析モジュール

景気サイクル判定、セクター別スコア補正、為替・金利感応度を
スクリーニング/スコアリングパイプラインに統合する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.sector.sector_config import (
    MarketPhase,
    SectorProfile,
    SectorType,
    SECTOR33_TO_TOPIX17,
    TOPIX17_PROFILES,
    get_sector_profile,
    get_topix17_sector,
)
from src.utils.logging_config import logger

if TYPE_CHECKING:
    from src.data.data_loader import MarketData


# ============================================================
# 景気サイクル判定
# ============================================================

def estimate_market_phase(macro_indicators: dict) -> MarketPhase:
    """マクロ指標から現在の景気サイクル局面を推定する

    判定ロジック:
    - 市場トレンド + 短期/中期モメンタムから4局面を推定
    - 金利動向（利上げ/利下げ）が加味可能な場合は補正

    Args:
        macro_indicators: MacroClient.fetch_indicators() の結果

    Returns:
        推定された景気サイクル局面
    """
    trend = macro_indicators.get("market_trend", "neutral")
    change_5d = macro_indicators.get("nikkei_change_5d", 0.0)
    change_20d = macro_indicators.get("nikkei_change_20d", 0.0)

    # 上昇トレンド + 中期モメンタム強い → 業績相場
    if trend == "bullish" and change_20d > 3.0:
        return MarketPhase.GYOSEKI

    # 上昇トレンド + 中期モメンタム弱い/回復初期 → 金融相場
    if trend == "bullish" and change_20d <= 3.0:
        return MarketPhase.KINYU

    # 下降トレンド + 短期急落 → 逆業績相場
    if trend == "bearish" and change_5d < -2.0:
        return MarketPhase.GYAKU_GYOSEKI

    # 下降トレンド + 緩やかな下落 → 逆金融相場
    if trend == "bearish":
        return MarketPhase.GYAKU_KINYU

    # 中立: 短期モメンタムで判定
    if change_5d > 1.0:
        return MarketPhase.KINYU
    if change_5d < -1.0:
        return MarketPhase.GYAKU_KINYU

    return MarketPhase.KINYU  # デフォルト


# ============================================================
# セクタースコア算出
# ============================================================

def calculate_sector_score(
    sector33_name: str,
    market_phase: MarketPhase,
    macro_indicators: dict | None = None,
) -> tuple[float, str]:
    """銘柄のセクターに基づくスコア補正値を計算する

    Args:
        sector33_name: 東証33業種名
        market_phase: 現在の景気サイクル局面
        macro_indicators: マクロ指標（為替情報等を含む場合）

    Returns:
        (score_adjustment, explanation)
        score_adjustment: -10.0 〜 +10.0 のスコア補正値
        explanation: 人間可読な説明文
    """
    topix17_name = get_topix17_sector(sector33_name)
    if topix17_name is None:
        return 0.0, ""

    profile = get_sector_profile(topix17_name)
    if profile is None:
        return 0.0, ""

    score = 0.0
    reasons = []

    # 1. 景気サイクルとの相性
    if market_phase in profile.favorable_phases:
        score += 5.0
        reasons.append(f"{market_phase.value}で有利")
    else:
        # ディフェンシブは不利局面でも大きく減点しない
        if profile.sector_type == SectorType.DEFENSIVE:
            score -= 1.0
        else:
            score -= 3.0

    # 2. セクタータイプ別の局面補正
    if market_phase == MarketPhase.GYAKU_GYOSEKI:
        if profile.sector_type == SectorType.DEFENSIVE:
            score += 3.0
            reasons.append("ディフェンシブ（逆業績相場で堅調）")
        elif profile.sector_type == SectorType.CYCLICAL:
            score -= 3.0

    if market_phase == MarketPhase.GYOSEKI:
        if profile.sector_type == SectorType.CYCLICAL:
            score += 2.0
            reasons.append("景気敏感（業績相場で上昇）")

    if market_phase == MarketPhase.KINYU:
        if profile.sector_type == SectorType.GROWTH:
            score += 3.0
            reasons.append("成長株（金融相場で有利）")
        if profile.sector_type == SectorType.FINANCIAL:
            # 金融相場初期は低金利 → 銀行にはまだ不利
            if profile.interest_rate_sensitivity > 0.5:
                score -= 1.0

    # スコアをクランプ
    score = max(-10.0, min(10.0, score))
    explanation = f"{topix17_name}: " + ", ".join(reasons) if reasons else topix17_name

    return round(score, 1), explanation


def resolve_sector_for_stock(
    code: str,
    stocks_df: pd.DataFrame,
) -> str | None:
    """銘柄コードから33業種名を取得する

    Args:
        code: 銘柄コード（4桁）
        stocks_df: 銘柄一覧DataFrame（Sector33CodeName列を含む）

    Returns:
        33業種名、取得できない場合はNone
    """
    if stocks_df.empty or "Code" not in stocks_df.columns:
        return None

    if "Sector33CodeName" not in stocks_df.columns:
        return None

    row = stocks_df[stocks_df["Code"] == code]
    if row.empty:
        return None

    return row.iloc[0].get("Sector33CodeName")
