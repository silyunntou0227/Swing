"""日本市場の休日判定モジュール"""

from __future__ import annotations

from datetime import date, timedelta


# 日本の祝日（固定日＋振替休日は毎年更新が必要）
# 2024-2026年分を定義。運用時は年次で更新する。
JAPANESE_HOLIDAYS = {
    # 2024
    date(2024, 1, 1), date(2024, 1, 8), date(2024, 2, 11),
    date(2024, 2, 12), date(2024, 2, 23), date(2024, 3, 20),
    date(2024, 4, 29), date(2024, 5, 3), date(2024, 5, 4),
    date(2024, 5, 5), date(2024, 5, 6), date(2024, 7, 15),
    date(2024, 8, 11), date(2024, 8, 12), date(2024, 9, 16),
    date(2024, 9, 22), date(2024, 9, 23), date(2024, 10, 14),
    date(2024, 11, 3), date(2024, 11, 4), date(2024, 11, 23),
    # 2025
    date(2025, 1, 1), date(2025, 1, 13), date(2025, 2, 11),
    date(2025, 2, 23), date(2025, 2, 24), date(2025, 3, 20),
    date(2025, 4, 29), date(2025, 5, 3), date(2025, 5, 4),
    date(2025, 5, 5), date(2025, 5, 6), date(2025, 7, 21),
    date(2025, 8, 11), date(2025, 9, 15), date(2025, 9, 23),
    date(2025, 10, 13), date(2025, 11, 3), date(2025, 11, 23),
    date(2025, 11, 24),
    # 2026
    date(2026, 1, 1), date(2026, 1, 12), date(2026, 2, 11),
    date(2026, 2, 23), date(2026, 3, 20), date(2026, 4, 29),
    date(2026, 5, 3), date(2026, 5, 4), date(2026, 5, 5),
    date(2026, 5, 6), date(2026, 7, 20), date(2026, 8, 11),
    date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23),
    date(2026, 10, 12), date(2026, 11, 3), date(2026, 11, 23),
}

# 年末年始休場（12/31〜1/3）
YEAR_END_HOLIDAYS_MD = [(12, 31), (1, 1), (1, 2), (1, 3)]


def is_market_holiday(d: date) -> bool:
    """指定日が東証の休場日かどうか判定"""
    # 土日
    if d.weekday() >= 5:
        return True
    # 祝日
    if d in JAPANESE_HOLIDAYS:
        return True
    # 年末年始
    if (d.month, d.day) in YEAR_END_HOLIDAYS_MD:
        return True
    return False


def is_trading_day(d: date) -> bool:
    """指定日が取引日かどうか"""
    return not is_market_holiday(d)


def get_last_trading_day(d: date | None = None) -> date:
    """直近の取引日を取得（当日が取引日ならそのまま返す）"""
    if d is None:
        d = date.today()
    while is_market_holiday(d):
        d -= timedelta(days=1)
    return d


def get_previous_trading_day(d: date) -> date:
    """前営業日を取得"""
    d -= timedelta(days=1)
    return get_last_trading_day(d)


def get_trading_days(start: date, end: date) -> list[date]:
    """期間内の取引日リストを返す"""
    days = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days
