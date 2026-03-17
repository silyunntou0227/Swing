"""TDnet（適時開示情報閲覧サービス）クライアント"""

from __future__ import annotations

import re
from datetime import date, datetime

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.utils.logging_config import logger

# TDnet の適時開示RSS
TDNET_RSS_URL = "https://www.release.tdnet.info/inbs/I_main_00.html"


class TDnetClient:
    """TDnet適時開示情報取得クライアント"""

    # 開示タイプの分類キーワード
    DISCLOSURE_TYPES = {
        "業績上方修正": ["上方修正", "業績予想の修正（増額）", "増額修正"],
        "業績下方修正": ["下方修正", "業績予想の修正（減額）", "減額修正"],
        "増配": ["増配", "配当予想の修正（増額）"],
        "減配": ["減配", "配当予想の修正（減額）", "無配"],
        "自社株買い": ["自己株式の取得", "自社株買い"],
        "株式分割": ["株式分割", "株式の分割"],
        "MBO": ["MBO", "マネジメント・バイアウト"],
        "TOB": ["公開買付け", "TOB"],
        "決算発表": ["決算短信", "四半期決算", "決算発表"],
    }

    def fetch_today_disclosures(self) -> pd.DataFrame:
        """当日の適時開示一覧を取得

        Returns:
            columns: datetime, code, company, title, disclosure_type, url
        """
        try:
            return self._fetch_from_html()
        except Exception as e:
            logger.warning(f"TDnet HTML取得失敗: {e}")
            return pd.DataFrame()

    def _fetch_from_html(self) -> pd.DataFrame:
        """TDnetのHTMLページから開示情報をスクレイピング"""
        resp = requests.get(TDNET_RSS_URL, timeout=15)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "lxml")

        records = []
        # 開示一覧テーブルからデータ抽出
        rows = soup.select("tr")
        for row in rows:
            cells = row.select("td")
            if len(cells) < 4:
                continue

            try:
                time_text = cells[0].get_text(strip=True)
                code_text = cells[1].get_text(strip=True)
                company = cells[2].get_text(strip=True)
                title = cells[3].get_text(strip=True)

                # 銘柄コード抽出（4桁数字）
                code_match = re.search(r"(\d{4})", code_text)
                if not code_match:
                    continue
                code = code_match.group(1)

                # リンク取得
                link_tag = cells[3].select_one("a")
                url = link_tag["href"] if link_tag and link_tag.get("href") else ""

                # 開示タイプ分類
                disclosure_type = self._classify_disclosure(title)

                records.append({
                    "datetime": time_text,
                    "code": code,
                    "company": company,
                    "title": title,
                    "disclosure_type": disclosure_type,
                    "url": url,
                })
            except (IndexError, KeyError):
                continue

        df = pd.DataFrame(records)
        logger.info(f"TDnet適時開示: {len(df)}件取得")
        return df

    def _classify_disclosure(self, title: str) -> str:
        """開示タイトルからタイプを分類"""
        for dtype, keywords in self.DISCLOSURE_TYPES.items():
            if any(kw in title for kw in keywords):
                return dtype
        return "その他"

    def get_disclosures_for_code(
        self, disclosures: pd.DataFrame, code: str
    ) -> pd.DataFrame:
        """特定銘柄の開示情報を抽出"""
        if disclosures.empty or "code" not in disclosures.columns:
            return pd.DataFrame()
        return disclosures[disclosures["code"] == code]
