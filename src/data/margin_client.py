"""信用残・空売りデータ取得クライアント"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.logging_config import logger

# 並列取得の設定
MARGIN_MAX_WORKERS = 5
MARGIN_REQUEST_TIMEOUT = 8


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
        return self._fetch_from_yahoo()

    def _fetch_from_yahoo(self) -> pd.DataFrame:
        """Yahoo Finance Japanから信用残情報を取得

        注: 全銘柄の一括取得は非効率なため、
        スクリーニング後の候補銘柄に対してのみ呼び出す想定
        """
        return pd.DataFrame()

    def fetch_margin_for_codes(self, codes: list[str]) -> pd.DataFrame:
        """指定銘柄の信用残情報を並列取得

        Args:
            codes: 銘柄コードリスト（4桁）

        Returns:
            columns: code, margin_buy, margin_sell, margin_ratio
        """
        if not codes:
            return pd.DataFrame()

        records = []
        with ThreadPoolExecutor(max_workers=MARGIN_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._fetch_single_margin, code): code
                for code in codes
            }
            for future in as_completed(futures):
                try:
                    info = future.result(timeout=MARGIN_REQUEST_TIMEOUT + 2)
                    if info:
                        records.append(info)
                except Exception as e:
                    code = futures[future]
                    logger.debug(f"信用残取得タイムアウト ({code}): {e}")

        return pd.DataFrame(records) if records else pd.DataFrame()

    def _fetch_single_margin(self, code: str) -> dict | None:
        """単一銘柄の信用残情報をYahoo Finance Japanから取得

        HTMLレイアウト変更に備え、複数のセレクタ戦略でパースする。
        """
        url = f"https://finance.yahoo.co.jp/quote/{code}.T/margin"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=MARGIN_REQUEST_TIMEOUT)
            resp.raise_for_status()

            # lxml が無ければ html.parser にフォールバック
            try:
                soup = BeautifulSoup(resp.text, "lxml")
            except Exception:
                soup = BeautifulSoup(resp.text, "html.parser")

            margin_buy = None
            margin_sell = None

            # 戦略1: テーブルからキーワード検索
            margin_buy, margin_sell = self._parse_from_tables(soup)

            # 戦略2: テーブルが見つからなければページ全体のテキストからパース
            if margin_buy is None or margin_sell is None:
                margin_buy, margin_sell = self._parse_from_text(soup)

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

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.debug(f"信用残ページなし ({code}): 404")
            else:
                logger.debug(f"信用残取得HTTPエラー ({code}): {e}")
            return None
        except requests.exceptions.Timeout:
            logger.debug(f"信用残取得タイムアウト ({code})")
            return None
        except (requests.exceptions.ConnectionError, ValueError) as e:
            logger.debug(f"信用残取得失敗 ({code}): {e}")
            return None

    @staticmethod
    def _parse_from_tables(soup: BeautifulSoup) -> tuple[int | None, int | None]:
        """テーブルから売残・買残を検索"""
        margin_buy = None
        margin_sell = None

        tables = soup.select("table")
        for table in tables:
            rows = table.select("tr")
            for row in rows:
                cells = row.select("td, th")
                texts = [c.get_text(strip=True) for c in cells]

                for i, text in enumerate(texts):
                    if "売残" in text and i + 1 < len(texts):
                        val = re.sub(r"[,株\s]", "", texts[i + 1])
                        try:
                            margin_sell = int(val)
                        except ValueError:
                            pass
                    elif "買残" in text and i + 1 < len(texts):
                        val = re.sub(r"[,株\s]", "", texts[i + 1])
                        try:
                            margin_buy = int(val)
                        except ValueError:
                            pass

        return margin_buy, margin_sell

    @staticmethod
    def _parse_from_text(soup: BeautifulSoup) -> tuple[int | None, int | None]:
        """ページテキスト全体から正規表現で売残・買残を抽出（フォールバック）"""
        margin_buy = None
        margin_sell = None

        full_text = soup.get_text()
        sell_match = re.search(r"売残[^\d]*([\d,]+)", full_text)
        buy_match = re.search(r"買残[^\d]*([\d,]+)", full_text)

        if sell_match:
            try:
                margin_sell = int(sell_match.group(1).replace(",", ""))
            except ValueError:
                pass
        if buy_match:
            try:
                margin_buy = int(buy_match.group(1).replace(",", ""))
            except ValueError:
                pass

        return margin_buy, margin_sell
