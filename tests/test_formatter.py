"""通知フォーマッターのユニットテスト"""

import datetime
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from src.notify.formatter import (
    LINEResultFormatter,
    ResultFormatter,
    ScoredCandidate,
    _build_reasoning_text,
    _chunked,
)
from src.main import _is_jpx_holiday


# ---------- ヘルパー ----------

def _make_candidate(**overrides) -> ScoredCandidate:
    """テスト用 ScoredCandidate を生成"""
    defaults = dict(
        code="7203", name="トヨタ自動車", close=2500.0,
        total_score=82.0, direction="buy",
        trend_score=80, macd_score=70, volume_score=60,
        fundamental_score=55, rsi_score=65, ichimoku_score=75,
        pattern_score=40, risk_reward_score=50,
        news_score=45, margin_score=30, sector_score=60,
        macro_adjustment=3,
        signals=["ゴールデンクロス", "MACD買い転換", "包み足(強気)"],
        per=15.2, pbr=1.8, roe=12.5,
        news_sentiment="ポジティブ",
        sector_name="電気機器", sector_explanation="半導体需要拡大",
    )
    defaults.update(overrides)
    return ScoredCandidate(**defaults)


# ---------- _chunked ----------

class TestChunked:
    def test_exact_division(self):
        assert _chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_remainder(self):
        assert _chunked([1, 2, 3, 4, 5], 3) == [[1, 2, 3], [4, 5]]

    def test_empty(self):
        assert _chunked([], 5) == []

    def test_single_chunk(self):
        assert _chunked([1, 2], 10) == [[1, 2]]


# ---------- _build_reasoning_text ----------

class TestBuildReasoningText:
    def test_contains_top3_factors(self):
        c = _make_candidate()
        text = _build_reasoning_text(c)
        assert "主要因子:" in text
        # 加重スコア上位3因子が含まれる
        assert "トレンド" in text

    def test_contains_signals(self):
        c = _make_candidate(signals=["ゴールデンクロス", "MACD買い転換"])
        text = _build_reasoning_text(c)
        assert "シグナル:" in text
        assert "ゴールデンクロス" in text
        assert "MACD買い転換" in text

    def test_signals_limited_to_4(self):
        c = _make_candidate(signals=[f"sig{i}" for i in range(10)])
        text = _build_reasoning_text(c)
        # 最大4つまで
        assert "sig3" in text
        assert "sig4" not in text

    def test_contains_fundamentals(self):
        c = _make_candidate(per=12.3, pbr=0.9, roe=15.0)
        text = _build_reasoning_text(c)
        assert "PER 12.3" in text
        assert "PBR 0.90" in text
        assert "ROE 15.0%" in text

    def test_no_fundamentals_when_none(self):
        c = _make_candidate(per=None, pbr=None, roe=None)
        text = _build_reasoning_text(c)
        assert "PER" not in text
        assert "PBR" not in text
        assert "ROE" not in text

    def test_contains_news_sentiment(self):
        c = _make_candidate(news_sentiment="ネガティブ")
        text = _build_reasoning_text(c)
        assert "ニュース: ネガティブ" in text

    def test_no_news_when_empty(self):
        c = _make_candidate(news_sentiment="")
        text = _build_reasoning_text(c)
        assert "ニュース:" not in text

    def test_contains_sector(self):
        c = _make_candidate(sector_name="電気機器", sector_explanation="半導体需要拡大")
        text = _build_reasoning_text(c)
        assert "セクター: 電気機器（半導体需要拡大）" in text

    def test_sector_without_explanation(self):
        c = _make_candidate(sector_name="電気機器", sector_explanation="")
        text = _build_reasoning_text(c)
        assert "セクター: 電気機器" in text
        assert "（" not in text

    def test_contains_macro_adjustment(self):
        c = _make_candidate(macro_adjustment=5)
        text = _build_reasoning_text(c)
        assert "マクロ調整: +5点" in text

    def test_negative_macro_adjustment(self):
        c = _make_candidate(macro_adjustment=-3)
        text = _build_reasoning_text(c)
        assert "マクロ調整: -3点" in text

    def test_no_macro_when_zero(self):
        c = _make_candidate(macro_adjustment=0)
        text = _build_reasoning_text(c)
        assert "マクロ調整" not in text

    def test_no_signals_when_empty(self):
        c = _make_candidate(signals=[])
        text = _build_reasoning_text(c)
        assert "シグナル:" not in text


# ---------- format_scoring_summary ----------

class TestFormatScoringSummary:
    def setup_method(self):
        self.formatter = ResultFormatter()

    def test_returns_list_of_embeds(self):
        buys = [_make_candidate(code="7203"), _make_candidate(code="6758")]
        embeds = self.formatter.format_scoring_summary(buys, [])
        assert isinstance(embeds, list)
        assert all(isinstance(e, dict) for e in embeds)

    def test_buy_only_has_ranking_and_reasoning(self):
        buys = [_make_candidate(code=f"{i}") for i in range(3)]
        embeds = self.formatter.format_scoring_summary(buys, [])
        # 1 ranking table + 1 reasoning embed (3 candidates < 5 per chunk)
        assert len(embeds) == 2
        assert "ランキング" in embeds[0]["title"]
        assert "推論サマリー" in embeds[1]["title"]

    def test_sell_only(self):
        sells = [_make_candidate(direction="sell", code=f"{i}") for i in range(2)]
        embeds = self.formatter.format_scoring_summary([], sells)
        assert len(embeds) == 2
        assert "売り候補ランキング" in embeds[0]["title"]
        assert "売り候補 推論サマリー" in embeds[1]["title"]

    def test_both_buy_and_sell(self):
        buys = [_make_candidate(code="7203")]
        sells = [_make_candidate(direction="sell", code="6758")]
        embeds = self.formatter.format_scoring_summary(buys, sells)
        # buy ranking + buy reasoning + sell ranking + sell reasoning
        assert len(embeds) == 4

    def test_empty_candidates(self):
        embeds = self.formatter.format_scoring_summary([], [])
        assert embeds == []

    def test_ranking_table_contains_code_and_score(self):
        buys = [_make_candidate(code="7203", total_score=85, close=3000)]
        embeds = self.formatter.format_scoring_summary(buys, [])
        desc = embeds[0]["description"]
        assert "7203" in desc
        assert "85" in desc
        assert "3,000" in desc

    def test_reasoning_fields_match_candidate_count(self):
        buys = [_make_candidate(code=f"{i}") for i in range(3)]
        embeds = self.formatter.format_scoring_summary(buys, [])
        reasoning_embed = embeds[1]
        assert len(reasoning_embed["fields"]) == 3

    def test_chunking_with_many_candidates(self):
        """6銘柄 → 5+1 の2チャンクに分割される"""
        buys = [_make_candidate(code=f"{i}") for i in range(6)]
        embeds = self.formatter.format_scoring_summary(buys, [])
        # 1 ranking + 2 reasoning embeds (5 + 1)
        assert len(embeds) == 3
        assert len(embeds[1]["fields"]) == 5
        assert len(embeds[2]["fields"]) == 1

    def test_ranking_table_color_buy(self):
        buys = [_make_candidate()]
        embeds = self.formatter.format_scoring_summary(buys, [])
        assert embeds[0]["color"] == ResultFormatter.COLOR_BUY

    def test_ranking_table_color_sell(self):
        sells = [_make_candidate(direction="sell")]
        embeds = self.formatter.format_scoring_summary([], sells)
        assert embeds[0]["color"] == ResultFormatter.COLOR_SELL

    def test_reasoning_field_name_includes_score(self):
        buys = [_make_candidate(name="テスト銘柄", code="1234", total_score=77)]
        embeds = self.formatter.format_scoring_summary(buys, [])
        field = embeds[1]["fields"][0]
        assert "テスト銘柄" in field["name"]
        assert "1234" in field["name"]
        assert "77" in field["name"]

    def test_long_name_truncated_in_table(self):
        """9文字以上の銘柄名はテーブルで8文字に切り詰め"""
        buys = [_make_candidate(name="あいうえおかきくけ")]  # 9文字
        embeds = self.formatter.format_scoring_summary(buys, [])
        desc = embeds[0]["description"]
        assert "あいうえおかきく" in desc  # 8文字に切り詰め
        assert "あいうえおかきくけ" not in desc


# ---------- _build_ranking_table (直接テスト) ----------

class TestBuildRankingTable:
    def setup_method(self):
        self.formatter = ResultFormatter()

    def test_code_block_formatting(self):
        candidates = [_make_candidate()]
        embed = self.formatter._build_ranking_table(candidates, "buy")
        assert embed["description"].startswith("```")
        assert embed["description"].endswith("```")

    def test_multiple_candidates_numbered(self):
        candidates = [
            _make_candidate(code="7203", total_score=90),
            _make_candidate(code="6758", total_score=80),
        ]
        embed = self.formatter._build_ranking_table(candidates, "buy")
        desc = embed["description"]
        assert " 1 " in desc
        assert " 2 " in desc
        assert "7203" in desc
        assert "6758" in desc

    def test_count_in_title(self):
        candidates = [_make_candidate() for _ in range(3)]
        embed = self.formatter._build_ranking_table(candidates, "sell")
        assert "3件" in embed["title"]


# ---------- LINEResultFormatter ----------


@dataclass
class _FakeMarketData:
    scan_date: str = "2026-03-18"
    stocks: list = field(default_factory=list)
    has_prices: bool = True
    has_financials: bool = True
    has_news: bool = False
    has_disclosures: bool = False
    prices: object = None
    disclosures: list = field(default_factory=list)
    news: list = field(default_factory=list)
    macro_indicators: dict = field(default_factory=dict)


class TestLINEResultFormatter:
    def setup_method(self):
        self.formatter = LINEResultFormatter()
        self.market_data = _FakeMarketData()

    def test_summary_with_candidates(self):
        buys = [_make_candidate(code="7203", total_score=82, stop_loss=2400, take_profit=2700, risk_reward_ratio=2.0)]
        text = self.formatter.format_summary(self.market_data, buys, [])
        assert "買い候補" in text
        assert "7203" in text
        assert "スコア82" in text
        assert "SL" in text

    def test_summary_no_candidates(self):
        text = self.formatter.format_summary(self.market_data, [], [])
        assert "条件を満たす候補銘柄がありませんでした" in text

    def test_summary_within_5000_chars(self):
        buys = [_make_candidate(code=f"{i:04d}", signals=[f"sig{j}" for j in range(10)]) for i in range(50)]
        text = self.formatter.format_summary(self.market_data, buys, [])
        assert len(text) <= 5000

    def test_sell_candidates_included(self):
        sells = [_make_candidate(code="6758", direction="sell")]
        text = self.formatter.format_summary(self.market_data, [], sells)
        assert "売り候補" in text
        assert "6758" in text


# ---------- Flex Message フォーマッター ----------


class TestLINEFlexMessage:
    def setup_method(self):
        self.formatter = LINEResultFormatter()
        self.market_data = _FakeMarketData()

    def test_flex_summary_returns_carousel(self):
        buys = [_make_candidate(code="7203")]
        result = self.formatter.build_flex_summary(self.market_data, buys, [])
        assert result["type"] == "carousel"
        assert isinstance(result["contents"], list)

    def test_flex_summary_header_bubble(self):
        buys = [_make_candidate(code="7203")]
        sells = [_make_candidate(code="6758", direction="sell")]
        result = self.formatter.build_flex_summary(self.market_data, buys, sells)
        header = result["contents"][0]
        assert header["type"] == "bubble"
        # ヘッダーに候補数が含まれる
        body_texts = [c.get("text", "") for c in header["body"]["contents"]
                      if c.get("type") == "text"]
        assert any("スキャン" in t for t in body_texts)

    def test_flex_summary_bubble_count(self):
        """1 header + 3 buy + 2 sell = 6 bubbles"""
        buys = [_make_candidate(code=f"{i}") for i in range(3)]
        sells = [_make_candidate(code=f"{i}", direction="sell") for i in range(2)]
        result = self.formatter.build_flex_summary(self.market_data, buys, sells)
        assert len(result["contents"]) == 6

    def test_flex_summary_max_12_bubbles(self):
        """carousel 上限12件"""
        buys = [_make_candidate(code=f"{i}") for i in range(15)]
        result = self.formatter.build_flex_summary(self.market_data, buys, [])
        assert len(result["contents"]) <= 12

    def test_flex_candidate_bubble_structure(self):
        c = _make_candidate(
            code="7203", name="トヨタ", total_score=85,
            stop_loss=2400, take_profit=2800, risk_reward_ratio=2.0,
        )
        bubble = self.formatter.build_flex_candidate(c, rank=1, direction="buy")
        assert bubble["type"] == "bubble"
        assert bubble["size"] == "kilo"
        assert "body" in bubble

    def test_flex_candidate_no_risk_when_zero(self):
        """stop_loss=0 のときリスク管理行が出ない"""
        c = _make_candidate(stop_loss=0, take_profit=0)
        bubble = self.formatter.build_flex_candidate(c, rank=1, direction="buy")
        body_str = str(bubble)
        assert "損切り" not in body_str

    def test_flex_candidate_shows_fundamentals(self):
        c = _make_candidate(per=12.0, pbr=0.9, roe=15.0)
        bubble = self.formatter.build_flex_candidate(c, rank=1, direction="buy")
        body_str = str(bubble)
        assert "PER 12.0" in body_str
        assert "PBR 0.90" in body_str
        assert "ROE 15.0%" in body_str

    def test_flex_sell_candidate_color(self):
        c = _make_candidate(direction="sell")
        bubble = self.formatter.build_flex_candidate(c, rank=1, direction="sell")
        body_str = str(bubble)
        assert LINEResultFormatter.COLOR_SELL in body_str


# ---------- _is_jpx_holiday ----------


class TestIsJpxHoliday:
    def test_saturday_is_holiday(self):
        assert _is_jpx_holiday(datetime.date(2026, 3, 14)) is True  # Saturday

    def test_sunday_is_holiday(self):
        assert _is_jpx_holiday(datetime.date(2026, 3, 15)) is True  # Sunday

    def test_weekday_is_not_holiday(self):
        # 月曜で祝日でなければ営業日
        assert _is_jpx_holiday(datetime.date(2026, 3, 16)) is False  # Monday

    def test_national_holiday(self):
        # 春分の日 2026-03-20 (金)
        assert _is_jpx_holiday(datetime.date(2026, 3, 20)) is True

    def test_new_year(self):
        assert _is_jpx_holiday(datetime.date(2026, 1, 1)) is True
