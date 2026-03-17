"""スコアリング・リスク管理のユニットテスト"""

import pytest

from src.config import SCORING_WEIGHTS


class TestScoringWeights:
    def test_weights_sum_to_one(self):
        assert SCORING_WEIGHTS.validate(), "スコアリングウェイトの合計が1.0ではありません"


class TestBalsaraRuinProbability:
    def test_positive_edge(self):
        from src.scoring.risk import calculate_balsara_ruin_probability

        # 勝率50%, 損益比2.0, リスク1% → 破産確率は低い
        prob = calculate_balsara_ruin_probability(
            win_rate=0.50, payoff_ratio=2.0, risk_per_trade=0.01
        )
        assert 0 <= prob <= 100
        assert prob < 10  # 期待値プラスなので低い

    def test_negative_edge(self):
        from src.scoring.risk import calculate_balsara_ruin_probability

        # 勝率30%, 損益比1.0 → 期待値マイナス → 破産確率100%
        prob = calculate_balsara_ruin_probability(
            win_rate=0.30, payoff_ratio=1.0, risk_per_trade=0.01
        )
        assert prob == 100.0

    def test_edge_cases(self):
        from src.scoring.risk import calculate_balsara_ruin_probability

        assert calculate_balsara_ruin_probability(win_rate=0.0) == 100.0
        assert calculate_balsara_ruin_probability(win_rate=1.0) == 0.0

    def test_high_risk(self):
        from src.scoring.risk import calculate_balsara_ruin_probability

        # 高リスク（10%/トレード）
        prob_high = calculate_balsara_ruin_probability(
            win_rate=0.50, payoff_ratio=2.0, risk_per_trade=0.10
        )
        # 低リスク（1%/トレード）
        prob_low = calculate_balsara_ruin_probability(
            win_rate=0.50, payoff_ratio=2.0, risk_per_trade=0.01
        )
        # 高リスクの方が破産確率が高い
        assert prob_high >= prob_low


class TestFibonacciLevels:
    def test_up_direction(self):
        from src.indicators.wave import calculate_fibonacci_levels

        levels = calculate_fibonacci_levels(1100, 1000, "up")
        # 上昇トレンドの押し目: 高値から下がるので全水準は1000-1100の間
        assert levels["fib_0.382"] < 1100
        assert levels["fib_0.382"] > 1000
        assert levels["fib_0.618"] < levels["fib_0.382"]

    def test_down_direction(self):
        from src.indicators.wave import calculate_fibonacci_levels

        levels = calculate_fibonacci_levels(1100, 1000, "down")
        # 下降トレンドの戻り: 安値から上がるので全水準は1000-1100の間
        assert levels["fib_0.382"] > 1000
        assert levels["fib_0.382"] < 1100
        assert levels["fib_0.618"] > levels["fib_0.382"]


class TestMarketCalendar:
    def test_is_trading_day(self):
        from datetime import date
        from src.data.market_calendar import is_trading_day, is_market_holiday

        # 平日
        monday = date(2024, 3, 4)
        assert is_trading_day(monday) is True

        # 土曜
        saturday = date(2024, 3, 2)
        assert is_market_holiday(saturday) is True

        # 元旦
        new_year = date(2024, 1, 1)
        assert is_market_holiday(new_year) is True

    def test_get_last_trading_day(self):
        from datetime import date
        from src.data.market_calendar import get_last_trading_day

        # 日曜 → 金曜を返す
        sunday = date(2024, 3, 3)
        last = get_last_trading_day(sunday)
        assert last.weekday() < 5  # 平日
        assert last <= sunday
