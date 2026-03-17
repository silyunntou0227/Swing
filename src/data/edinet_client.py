"""EDINET API クライアント（金融庁 電子開示システム）"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import requests

from src.config import EDINET_API_KEY
from src.utils.logging_config import logger

EDINET_API_BASE = "https://api.edinet-fsa.go.jp/api/v2"


class EDINETClient:
    """EDINET API クライアント"""

    # 書類タイプコード
    DOC_TYPE_SECURITIES_REPORT = "120"       # 有価証券報告書
    DOC_TYPE_QUARTERLY_REPORT = "140"        # 四半期報告書
    DOC_TYPE_LARGE_HOLDING = "160"           # 大量保有報告書
    DOC_TYPE_LARGE_HOLDING_CHANGE = "170"    # 変更報告書

    def __init__(self) -> None:
        self._api_key = EDINET_API_KEY
        if not self._api_key:
            logger.warning("EDINET_API_KEY が未設定です")

    def fetch_recent_filings(self, days_back: int = 3) -> pd.DataFrame:
        """直近N日間の開示書類一覧を取得

        Args:
            days_back: 遡る日数

        Returns:
            columns: date, doc_id, edinet_code, sec_code, filer_name,
                     doc_type, doc_description, filing_type
        """
        if not self._api_key:
            return pd.DataFrame()

        all_records = []
        today = date.today()

        for i in range(days_back):
            target_date = today - timedelta(days=i)
            records = self._fetch_filing_list(target_date)
            all_records.extend(records)

        df = pd.DataFrame(all_records)
        if not df.empty:
            # 銘柄コード正規化（4桁）
            if "sec_code" in df.columns:
                df["code"] = df["sec_code"].astype(str).str[:4]
        return df

    def _fetch_filing_list(self, target_date: date) -> list[dict]:
        """指定日の書類一覧を取得"""
        url = f"{EDINET_API_BASE}/documents.json"
        params = {
            "date": target_date.strftime("%Y-%m-%d"),
            "type": 2,  # 2=書類一覧+メタデータ
            "Subscription-Key": self._api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"EDINET API エラー ({target_date}): {e}")
            return []

        results = data.get("results", [])
        records = []
        for item in results:
            # 株式関連の書類のみ抽出
            sec_code = item.get("secCode", "")
            if not sec_code or not re.match(r"\d{5}", str(sec_code)):
                continue

            doc_type_code = item.get("docTypeCode", "")
            filing_type = self._classify_filing(doc_type_code)

            records.append({
                "date": target_date.isoformat(),
                "doc_id": item.get("docID", ""),
                "edinet_code": item.get("edinetCode", ""),
                "sec_code": sec_code,
                "filer_name": item.get("filerName", ""),
                "doc_type": doc_type_code,
                "doc_description": item.get("docDescription", ""),
                "filing_type": filing_type,
            })

        return records

    def _classify_filing(self, doc_type_code: str) -> str:
        """書類タイプコードから分類"""
        mapping = {
            self.DOC_TYPE_SECURITIES_REPORT: "有価証券報告書",
            self.DOC_TYPE_QUARTERLY_REPORT: "四半期報告書",
            self.DOC_TYPE_LARGE_HOLDING: "大量保有報告書_新規",
            self.DOC_TYPE_LARGE_HOLDING_CHANGE: "大量保有報告書_変更",
        }
        return mapping.get(doc_type_code, "その他")

    def get_large_holdings_for_code(
        self, filings: pd.DataFrame, code: str
    ) -> pd.DataFrame:
        """特定銘柄の大量保有報告書を抽出"""
        if filings.empty or "code" not in filings.columns:
            return pd.DataFrame()
        mask = (filings["code"] == code) & (
            filings["filing_type"].str.startswith("大量保有")
        )
        return filings[mask]
