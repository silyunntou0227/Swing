"""スクリーニングのユニットテスト"""

import pandas as pd
import pytest

from src.config import (
    PER_MIN, PER_MAX, PBR_MIN, PBR_MAX,
    VOLUME_AVG_MIN, TURNOVER_MIN,
)


class TestFundamentalFilter:
    def test_per_filter(self):
        from src.screening.fundamental import filter_fundamentals

        stocks = pd.DataFrame({"Code": ["1001", "1002", "1003"]})
        financials = pd.DataFrame({
            "Code": ["1001", "1002", "1003"],
            "EarningsPerShare": [100, 10, 50],
        })

        result = filter_fundamentals(stocks, financials)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(stocks)

    def test_empty_financials(self):
        from src.screening.fundamental import filter_fundamentals

        stocks = pd.DataFrame({"Code": ["1001", "1002"]})
        result = filter_fundamentals(stocks, pd.DataFrame())
        assert len(result) == 2  # 財務データなし → 全銘柄通過


class TestLiquidityFilter:
    def test_filter_liquidity(self):
        from src.screening.liquidity import filter_liquidity
        import numpy as np

        codes = ["1001", "1002"]
        dates = pd.date_range("2024-01-01", periods=30, freq="B")

        prices_data = []
        for code in codes:
            for d in dates:
                vol = 100000 if code == "1001" else 1000  # 1002は低流動性
                prices_data.append({
                    "Code": code,
                    "Date": d,
                    "Close": 1000,
                    "Volume": vol,
                })

        prices = pd.DataFrame(prices_data)
        result = filter_liquidity(codes, prices)

        # 1001は通過、1002は流動性不足で除外
        assert "1001" in result
        assert "1002" not in result


class TestNewsFilter:
    def test_should_exclude_mbo(self):
        from src.screening.news_filter import NewsFilter
        from src.data.data_loader import MarketData

        nf = NewsFilter()
        market_data = MarketData()
        market_data.disclosures = pd.DataFrame({
            "code": ["1001"],
            "disclosure_type": ["MBO"],
            "title": ["MBO実施のお知らせ"],
        })

        assert nf.should_exclude("1001", market_data) is True
        assert nf.should_exclude("1002", market_data) is False

    def test_disclosure_score(self):
        from src.screening.news_filter import NewsFilter
        from src.data.data_loader import MarketData

        nf = NewsFilter()
        market_data = MarketData()
        market_data.disclosures = pd.DataFrame({
            "code": ["1001"],
            "disclosure_type": ["業績上方修正"],
            "title": ["業績予想の修正（増額）"],
        })

        score, summary = nf.calculate_disclosure_score("1001", market_data)
        assert score > 0
        assert "上方修正" in summary
