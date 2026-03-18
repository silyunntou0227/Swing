"""セクター分析のユニットテスト"""

import pandas as pd
import pytest

from src.sector.sector_config import (
    MarketPhase,
    SectorType,
    SECTOR33_TO_TOPIX17,
    TOPIX17_PROFILES,
    get_topix17_sector,
    get_sector_profile,
)
from src.sector.sector_analyzer import (
    estimate_market_phase,
    calculate_sector_score,
    resolve_sector_for_stock,
)


class TestSectorConfig:
    def test_all_topix17_have_profiles(self):
        """TOPIX-17マッピング先すべてにプロファイルが定義されている"""
        for sector33, topix17 in SECTOR33_TO_TOPIX17.items():
            assert topix17 in TOPIX17_PROFILES, (
                f"{sector33} → {topix17} のプロファイルが未定義"
            )

    def test_get_topix17_sector(self):
        assert get_topix17_sector("銀行業") == "銀行"
        assert get_topix17_sector("電気機器") == "電気・精密"
        assert get_topix17_sector("食料品") == "食品"
        assert get_topix17_sector("存在しない業種") is None

    def test_get_sector_profile(self):
        profile = get_sector_profile("銀行")
        assert profile is not None
        assert profile.sector_type == SectorType.FINANCIAL
        assert profile.interest_rate_sensitivity > 0.5  # 金利上昇で好影響

    def test_defensive_sectors_low_beta(self):
        """ディフェンシブセクターはβ < 1.0"""
        for name, profile in TOPIX17_PROFILES.items():
            if profile.sector_type == SectorType.DEFENSIVE:
                assert profile.beta < 1.0, f"{name} はディフェンシブだがβ={profile.beta}"

    def test_sector33_mapping_completeness(self):
        """主要33業種がマッピングに含まれている"""
        expected = [
            "銀行業", "電気機器", "輸送用機器", "食料品",
            "医薬品", "建設業", "不動産業", "情報・通信業",
        ]
        for sector in expected:
            assert sector in SECTOR33_TO_TOPIX17, f"{sector} がマッピングにない"


class TestMarketPhaseEstimation:
    def test_bullish_strong_momentum(self):
        """強い上昇トレンド → 業績相場"""
        macro = {
            "market_trend": "bullish",
            "nikkei_change_5d": 3.0,
            "nikkei_change_20d": 8.0,
        }
        assert estimate_market_phase(macro) == MarketPhase.GYOSEKI

    def test_bullish_weak_momentum(self):
        """弱い上昇トレンド → 金融相場"""
        macro = {
            "market_trend": "bullish",
            "nikkei_change_5d": 1.0,
            "nikkei_change_20d": 1.5,
        }
        assert estimate_market_phase(macro) == MarketPhase.KINYU

    def test_bearish_sharp_decline(self):
        """急落 → 逆業績相場"""
        macro = {
            "market_trend": "bearish",
            "nikkei_change_5d": -4.0,
            "nikkei_change_20d": -8.0,
        }
        assert estimate_market_phase(macro) == MarketPhase.GYAKU_GYOSEKI

    def test_bearish_gradual_decline(self):
        """緩やかな下落 → 逆金融相場"""
        macro = {
            "market_trend": "bearish",
            "nikkei_change_5d": -0.5,
            "nikkei_change_20d": -3.0,
        }
        assert estimate_market_phase(macro) == MarketPhase.GYAKU_KINYU

    def test_empty_macro(self):
        """マクロ指標なし → デフォルト（金融相場）"""
        phase = estimate_market_phase({})
        assert phase == MarketPhase.KINYU


class TestSectorScoring:
    def test_bank_in_rising_rate(self):
        """銀行セクターは逆金融相場（金利上昇局面）で有利"""
        score, explanation = calculate_sector_score(
            "銀行業", MarketPhase.GYAKU_KINYU
        )
        assert score > 0, f"銀行は逆金融相場でプラスのはず: {score}"
        assert "銀行" in explanation

    def test_defensive_in_recession(self):
        """食品セクターは逆業績相場で有利"""
        score, explanation = calculate_sector_score(
            "食料品", MarketPhase.GYAKU_GYOSEKI
        )
        assert score > 0, f"食品は逆業績相場でプラスのはず: {score}"

    def test_cyclical_in_recession(self):
        """景気敏感セクターは逆業績相場で不利"""
        score, _ = calculate_sector_score(
            "機械", MarketPhase.GYAKU_GYOSEKI
        )
        assert score < 0, f"機械は逆業績相場でマイナスのはず: {score}"

    def test_growth_in_kinyu(self):
        """成長株は金融相場で有利"""
        score, _ = calculate_sector_score(
            "情報・通信業", MarketPhase.KINYU
        )
        assert score > 0

    def test_unknown_sector(self):
        """未知のセクターはスコア0"""
        score, explanation = calculate_sector_score(
            "未知の業種", MarketPhase.KINYU
        )
        assert score == 0.0
        assert explanation == ""

    def test_score_range(self):
        """全セクター・全局面でスコアが-10〜+10の範囲"""
        for sector33 in SECTOR33_TO_TOPIX17:
            for phase in MarketPhase:
                score, _ = calculate_sector_score(sector33, phase)
                assert -10.0 <= score <= 10.0, (
                    f"{sector33}/{phase}: score={score} が範囲外"
                )


class TestResolveSector:
    def test_resolve_with_valid_data(self):
        stocks = pd.DataFrame({
            "Code": ["7203", "8306"],
            "Sector33CodeName": ["輸送用機器", "銀行業"],
        })
        assert resolve_sector_for_stock("7203", stocks) == "輸送用機器"
        assert resolve_sector_for_stock("8306", stocks) == "銀行業"

    def test_resolve_missing_code(self):
        stocks = pd.DataFrame({
            "Code": ["7203"],
            "Sector33CodeName": ["輸送用機器"],
        })
        assert resolve_sector_for_stock("9999", stocks) is None

    def test_resolve_empty_df(self):
        assert resolve_sector_for_stock("7203", pd.DataFrame()) is None
