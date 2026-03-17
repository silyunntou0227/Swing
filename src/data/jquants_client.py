"""J-Quants API V2 ラッパーモジュール（APIキー認証対応）"""

from __future__ import annotations

import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests

from src.config import JQUANTS_API_KEY, HISTORY_DAYS
from src.data.market_calendar import get_last_trading_day
from src.utils.logging_config import logger

BASE_URL = "https://api.jquants.com/v2"


class JQuantsClient:
    """J-Quants API V2のラッパークラス（APIキー認証）"""

    def __init__(self) -> None:
        if not JQUANTS_API_KEY:
            raise ValueError(
                "JQUANTS_API_KEY を環境変数に設定してください"
            )
        self._headers = {"x-api-key": JQUANTS_API_KEY}
        logger.info("J-Quants API V2 クライアント初期化完了")

    def _get(self, endpoint: str, params: dict | None = None) -> requests.Response:
        """GETリクエスト（レートリミット対応）"""
        url = f"{BASE_URL}{endpoint}"
        resp = requests.get(url, headers=self._headers, params=params, timeout=60)
        resp.raise_for_status()
        return resp

    def fetch_listed_stocks(self) -> pd.DataFrame:
        """全上場銘柄一覧を取得"""
        logger.info("上場銘柄一覧を取得中...")
        resp = self._get("/equities/listed")
        data = resp.json()
        df = pd.DataFrame(data.get("listed", data.get("info", [])))
        logger.info(f"上場銘柄一覧: {len(df)}件取得")
        return df

    def fetch_daily_quotes(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """全銘柄の日足株価データを取得

        Args:
            start_date: 取得開始日（デフォルト: HISTORY_DAYS営業日前）
            end_date: 取得終了日（デフォルト: 直近営業日）
        """
        if end_date is None:
            end_date = get_last_trading_day()
        if start_date is None:
            start_date = end_date - timedelta(days=int(HISTORY_DAYS * 1.5))

        logger.info(f"日足株価データ取得中: {start_date} 〜 {end_date}")

        all_frames = []
        current = start_date
        while current <= end_date:
            params = {"date": current.strftime("%Y-%m-%d")}
            try:
                resp = self._get("/equities/bars/daily", params=params)
                data = resp.json()
                rows = data.get("daily_quotes", data.get("bars", []))
                if rows:
                    all_frames.append(pd.DataFrame(rows))
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    pass  # 休日等でデータなし
                else:
                    logger.warning(f"株価取得エラー ({current}): {e}")
            current += timedelta(days=1)
            time.sleep(0.3)  # レートリミット対策

        if not all_frames:
            logger.warning("株価データが取得できませんでした")
            return pd.DataFrame()

        df = pd.concat(all_frames, ignore_index=True)

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        elif "date" in df.columns:
            df["Date"] = pd.to_datetime(df["date"])

        logger.info(f"日足株価データ: {len(df)}行取得（{df['Code'].nunique() if 'Code' in df.columns else '?'}銘柄）")
        return df

    def fetch_financial_data(self) -> pd.DataFrame:
        """全銘柄の財務データを取得"""
        logger.info("財務データ取得中...")
        resp = self._get("/fins/statements")
        data = resp.json()
        df = pd.DataFrame(data.get("statements", data.get("fins_statements", [])))
        logger.info(f"財務データ: {len(df)}行取得")
        return df

    def fetch_financial_announcement(self) -> pd.DataFrame:
        """決算発表スケジュールを取得"""
        logger.info("決算発表スケジュール取得中...")
        try:
            resp = self._get("/fins/announcement")
            data = resp.json()
            df = pd.DataFrame(data.get("announcement", []))
            logger.info(f"決算発表スケジュール: {len(df)}件取得")
            return df
        except Exception as e:
            logger.warning(f"決算発表スケジュール取得失敗: {e}")
            return pd.DataFrame()
