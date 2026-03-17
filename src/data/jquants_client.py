"""J-Quants API V2 ラッパーモジュール（APIキー認証対応）

無料プランの厳しい制限（5 req/min, 一部エンドポイント403）を考慮し、
補助データソースとしての利用に限定する設計。
主力の株価データは yfinance 経由で取得する。
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.config import JQUANTS_API_KEY
from src.data.market_calendar import get_last_trading_day
from src.utils.logging_config import logger

BASE_URL = "https://api.jquants.com/v2"

# 無料プラン: 5 req/min → 12.5秒間隔で安全にリクエスト
THROTTLE_SECONDS = 12.5


class JQuantsClient:
    """J-Quants API V2のラッパークラス（APIキー認証）

    無料プランでは以下の制限がある:
    - /equities/listed → 403 (Forbidden)
    - /fins/statements → 403 (Forbidden)
    - /equities/bars/daily → 5 req/min（超過で429 + 5分遮断）
    """

    def __init__(self) -> None:
        if not JQUANTS_API_KEY:
            raise ValueError(
                "JQUANTS_API_KEY を環境変数に設定してください"
            )
        self._headers = {"x-api-key": JQUANTS_API_KEY}
        self._last_request_time = 0.0
        logger.info("J-Quants API V2 クライアント初期化完了")

    def _throttle(self) -> None:
        """レートリミット対策: 前回リクエストから12.5秒以上空ける"""
        elapsed = time.time() - self._last_request_time
        if elapsed < THROTTLE_SECONDS:
            wait = THROTTLE_SECONDS - elapsed
            logger.debug(f"J-Quants スロットリング: {wait:.1f}秒待機")
            time.sleep(wait)

    def _get(self, endpoint: str, params: dict | None = None) -> requests.Response:
        """GETリクエスト（スロットリング + リトライ対応）"""
        self._throttle()
        url = f"{BASE_URL}{endpoint}"
        self._last_request_time = time.time()

        resp = requests.get(url, headers=self._headers, params=params, timeout=60)

        # 429 Too Many Requests → バックオフして1回リトライ
        if resp.status_code == 429:
            logger.warning("J-Quants 429 レートリミット検出 → 60秒待機してリトライ")
            time.sleep(60)
            self._last_request_time = time.time()
            resp = requests.get(url, headers=self._headers, params=params, timeout=60)

        resp.raise_for_status()
        return resp

    def fetch_listed_stocks(self) -> pd.DataFrame:
        """全上場銘柄一覧を取得（無料プランでは403になる可能性大）"""
        logger.info("J-Quants: 上場銘柄一覧を取得中...")
        try:
            resp = self._get("/equities/listed")
            data = resp.json()
            df = pd.DataFrame(data.get("listed", data.get("info", [])))
            logger.info(f"J-Quants: 上場銘柄一覧: {len(df)}件取得")
            return df
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.warning("J-Quants: 上場銘柄一覧は無料プランでは利用不可（403）")
            else:
                logger.warning(f"J-Quants: 上場銘柄一覧取得失敗: {e}")
            return pd.DataFrame()

    def fetch_daily_quotes_by_date(
        self,
        target_date: date,
    ) -> pd.DataFrame:
        """特定の日付の全銘柄株価を一括取得（1リクエスト）

        レポート推奨の「日付ベース一括取得」方式。
        1リクエストで当該日の全銘柄（~4000件）のOHLCVを取得できる。
        """
        params = {"date": target_date.strftime("%Y-%m-%d")}
        try:
            resp = self._get("/equities/bars/daily", params=params)
            data = resp.json()
            rows = data.get("daily_quotes", data.get("bars", []))
            if rows:
                df = pd.DataFrame(rows)
                logger.info(f"J-Quants: {target_date} → {len(df)}行取得")
                return df
            return pd.DataFrame()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (403, 404):
                logger.debug(f"J-Quants: {target_date} データなし ({e.response.status_code})")
            else:
                logger.warning(f"J-Quants: 株価取得エラー ({target_date}): {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"J-Quants: 株価取得エラー ({target_date}): {e}")
            return pd.DataFrame()

    def fetch_financial_data(self) -> pd.DataFrame:
        """全銘柄の財務データを取得（無料プランでは403になる可能性大）"""
        logger.info("J-Quants: 財務データ取得中...")
        try:
            resp = self._get("/fins/statements")
            data = resp.json()
            df = pd.DataFrame(data.get("statements", data.get("fins_statements", [])))
            logger.info(f"J-Quants: 財務データ: {len(df)}行取得")
            return df
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.warning("J-Quants: 財務データは無料プランでは利用不可（403）")
            else:
                logger.warning(f"J-Quants: 財務データ取得失敗: {e}")
            return pd.DataFrame()

    def fetch_financial_announcement(self) -> pd.DataFrame:
        """決算発表スケジュールを取得"""
        logger.info("J-Quants: 決算発表スケジュール取得中...")
        try:
            resp = self._get("/fins/announcement")
            data = resp.json()
            df = pd.DataFrame(data.get("announcement", []))
            logger.info(f"J-Quants: 決算発表スケジュール: {len(df)}件取得")
            return df
        except Exception as e:
            logger.warning(f"J-Quants: 決算発表スケジュール取得失敗: {e}")
            return pd.DataFrame()
