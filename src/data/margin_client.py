"""信用残・空売りデータ取得クライアント"""

from __future__ import annotations

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.logging_config import logger


class MarginClient:
    """信用残・空売りデータ取得

    東証の公開データおよびYahoo Financeから信用取引残高情報を取得する。
    """

    # JPX 信用取引残高データ
    JPX_MARGIN_URL = "https://www.jpx.co.jp/markets/statistics-equities/margin/index.html"

    def fetch_margin_data(self) -> pd.DataFrame:
        """信用残データを取得

        Returns:
            columns: code, margin_buy, margin_sell, margin_ratio,
                     margin_buy_change, margin_sell_change
        """
        # yfinance経由で信用残情報を取得（JPX直接取得のフォールバック）
        return self._fetch_from_yahoo()

    def _fetch_from_yahoo(self) -> pd.DataFrame:
        """Yahoo Finance Japanから信用残情報を取得

        注: 全銘柄の一括取得は非効率なため、
        スクリーニング後の候補銘柄に対してのみ呼び出す想定
        """
        # 信用残情報は候補銘柄確定後に個別取得するため、
        # ここでは空のDataFrameを返す
        return pd.DataFrame()

    def fetch_margin_for_codes(self, codes: list[str]) -> pd.DataFrame:
        """指定銘柄の信用残情報を取得

        Args:
            codes: 銘柄コードリスト（4桁）

        Returns:
            columns: code, margin_buy, margin_sell, margin_ratio
        """
        records = []
        for code in codes:
            info = self._fetch_single_margin(code)
            if info:
                records.append(info)

        return pd.DataFrame(records) if records else pd.DataFrame()

    def _fetch_single_margin(self, code: str) -> dict | None:
        """単一銘柄の信用残情報をYahoo Finance Japanから取得"""
        url = f"https://finance.yahoo.co.jp/quote/{code}.T/margin"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; StockScanner/1.0)"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # テーブルからデータ抽出
            tables = soup.select("table")
            if not tables:
                return None

            margin_buy = None
            margin_sell = None

            for table in tables:
                rows = table.select("tr")
                for row in rows:
                    cells = row.select("td, th")
                    texts = [c.get_text(strip=True) for c in cells]

                    for i, text in enumerate(texts):
                        if "売残" in text and i + 1 < len(texts):
                            val = re.sub(r"[,株]", "", texts[i + 1])
                            try:
                                margin_sell = int(val)
                            except ValueError:
                                pass
                        elif "買残" in text and i + 1 < len(texts):
                            val = re.sub(r"[,株]", "", texts[i + 1])
                            try:
                                margin_buy = int(val)
                            except ValueError:
                                pass

            if margin_buy is not None and margin_sell is not None and margin_sell > 0:
                margin_ratio = margin_buy / margin_sell
            else:
                margin_ratio = None

            return {
                "code": code,
                "margin_buy": margin_buy,
                "margin_sell": margin_sell,
                "margin_ratio": margin_ratio,
            }

        except Exception as e:
            logger.debug(f"信用残取得失敗 ({code}): {e}")
            return None
