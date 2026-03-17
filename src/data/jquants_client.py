"""J-Quants API V2 ラッパーモジュール（バルクCSVダウンロード対応）"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

try:
    import jquantsapi
except ImportError:
    jquantsapi = None  # type: ignore[assignment]

from src.config import JQUANTS_MAIL, JQUANTS_PASSWORD, HISTORY_DAYS
from src.data.market_calendar import get_last_trading_day, get_trading_days
from src.utils.logging_config import logger


class JQuantsClient:
    """J-Quants API V2のラッパークラス"""

    def __init__(self) -> None:
        if jquantsapi is None:
            raise ImportError("jquants-api-client がインストールされていません")
        if not JQUANTS_MAIL or not JQUANTS_PASSWORD:
            raise ValueError(
                "JQUANTS_MAIL と JQUANTS_PASSWORD を環境変数に設定してください"
            )
        self._client = jquantsapi.Client(
            mail_address=JQUANTS_MAIL,
            password=JQUANTS_PASSWORD,
        )
        logger.info("J-Quants API クライアント初期化完了")

    def fetch_listed_stocks(self) -> pd.DataFrame:
        """全上場銘柄一覧を取得

        Returns:
            columns: Code, CompanyName, CompanyNameEnglish, Sector17Code,
                     Sector33Code, ScaleCategory, MarketCode, MarketCodeName, ...
        """
        logger.info("上場銘柄一覧を取得中...")
        df = self._client.get_listed_info()
        logger.info(f"上場銘柄一覧: {len(df)}件取得")
        return df

    def fetch_daily_quotes(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """全銘柄の日足株価データを取得（バルク）

        Args:
            start_date: 取得開始日（デフォルト: HISTORY_DAYS営業日前）
            end_date: 取得終了日（デフォルト: 直近営業日）

        Returns:
            columns: Date, Code, Open, High, Low, Close, Volume,
                     TurnoverValue, AdjustmentFactor, AdjustmentOpen,
                     AdjustmentHigh, AdjustmentLow, AdjustmentClose, AdjustmentVolume
        """
        if end_date is None:
            end_date = get_last_trading_day()
        if start_date is None:
            start_date = end_date - timedelta(days=int(HISTORY_DAYS * 1.5))

        logger.info(f"日足株価データ取得中: {start_date} 〜 {end_date}")
        df = self._client.get_prices_daily_quotes(
            date_yyyymmdd=start_date.strftime("%Y%m%d"),
        )

        # 日付範囲でフィルタ
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[
                (df["Date"] >= pd.Timestamp(start_date))
                & (df["Date"] <= pd.Timestamp(end_date))
            ]

        logger.info(f"日足株価データ: {len(df)}行取得（{df['Code'].nunique()}銘柄）")
        return df

    def fetch_financial_data(self) -> pd.DataFrame:
        """全銘柄の財務データを取得

        Returns:
            columns: DisclosedDate, DisclosedTime, LocalCode,
                     EarningsPerShare, BookValuePerShare, ResultDividendPerShare,
                     Profit, ...
        """
        logger.info("財務データ取得中...")
        df = self._client.get_fins_statements()
        logger.info(f"財務データ: {len(df)}行取得")
        return df

    def fetch_financial_announcement(self) -> pd.DataFrame:
        """決算発表スケジュールを取得"""
        logger.info("決算発表スケジュール取得中...")
        try:
            df = self._client.get_fins_announcement()
            logger.info(f"決算発表スケジュール: {len(df)}件取得")
            return df
        except Exception as e:
            logger.warning(f"決算発表スケジュール取得失敗: {e}")
            return pd.DataFrame()
