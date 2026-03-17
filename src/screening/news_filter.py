"""ニュース・開示情報フィルタ＆スコアリングモジュール（Layer 4）"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.config import (
    DISCLOSURE_SCORES,
    EDINET_SCORES,
    POSITIVE_KEYWORDS,
    NEGATIVE_KEYWORDS,
    NEWS_SCORE_MAX,
    NEWS_SCORE_MIN,
)
from src.data.news_client import NewsClient
from src.utils.logging_config import logger

if TYPE_CHECKING:
    from src.data.data_loader import MarketData


class NewsFilter:
    """ニュース・開示情報に基づくフィルタリングとスコアリング"""

    def __init__(self) -> None:
        self._news_client = NewsClient()

    def should_exclude(self, code: str, market_data: MarketData) -> bool:
        """銘柄を除外すべきか判定（MBO/TOB等）

        Args:
            code: 銘柄コード
            market_data: 全データ

        Returns:
            True = 除外すべき
        """
        if market_data.has_disclosures:
            code_disclosures = market_data.disclosures[
                market_data.disclosures.get("code", pd.Series()) == code
            ]
            if not code_disclosures.empty:
                for _, row in code_disclosures.iterrows():
                    dtype = row.get("disclosure_type", "")
                    score = DISCLOSURE_SCORES.get(dtype, 0)
                    if score == -999:
                        logger.debug(f"{code}: {dtype} により除外")
                        return True
        return False

    def calculate_disclosure_score(
        self, code: str, market_data: MarketData
    ) -> tuple[float, str]:
        """適時開示スコアを計算

        Returns:
            (score, summary_text)
        """
        score = 0.0
        summaries = []

        # TDnet適時開示
        if market_data.has_disclosures and "code" in market_data.disclosures.columns:
            code_disclosures = market_data.disclosures[
                market_data.disclosures["code"] == code
            ]
            for _, row in code_disclosures.iterrows():
                dtype = row.get("disclosure_type", "その他")
                title = row.get("title", "")
                dscore = DISCLOSURE_SCORES.get(dtype, 0)
                if dscore != -999:  # 除外対象は別処理
                    score += dscore
                    summaries.append(f"{dtype}({dscore:+d})")

        # EDINET大量保有報告書
        if not market_data.edinet_filings.empty and "code" in market_data.edinet_filings.columns:
            code_filings = market_data.edinet_filings[
                market_data.edinet_filings["code"] == code
            ]
            for _, row in code_filings.iterrows():
                ftype = row.get("filing_type", "")
                escore = EDINET_SCORES.get(ftype, 0)
                if escore != 0:
                    score += escore
                    summaries.append(f"{ftype}({escore:+d})")

        summary = ", ".join(summaries) if summaries else ""
        return score, summary

    def calculate_news_sentiment(
        self,
        code: str,
        company_name: str,
        market_data: MarketData,
    ) -> tuple[float, str, str]:
        """ニュースセンチメントスコアを計算

        Args:
            code: 銘柄コード
            company_name: 企業名（ニュース検索用）
            market_data: 全データ

        Returns:
            (sentiment_score, sentiment_label, summary_text)
        """
        # 市場全体のニュースからセンチメントを取得
        all_news_score = 0.0
        if market_data.has_news:
            all_news_score = self._score_news_dataframe(
                market_data.news, keywords_filter=None
            )

        # 個別銘柄のニュース: まず取得済みニュースから検索（追加API不要）
        # マッチしなかった場合のみAPIを呼ぶ（候補は最大15件なので負荷は限定的）
        company_score = 0.0
        company_summary = ""
        try:
            search_name = company_name[:10] if len(company_name) > 10 else company_name
            if market_data.has_news and search_name:
                # 取得済みニュースから銘柄名でフィルタ（API呼び出しなし）
                company_score = self._score_news_dataframe(
                    market_data.news, keywords_filter=search_name
                )
                matched = market_data.news[
                    market_data.news.apply(
                        lambda r: search_name in str(r.get("title", ""))
                        + str(r.get("description", "")),
                        axis=1,
                    )
                ]
                if not matched.empty and "title" in matched.columns:
                    company_summary = matched.iloc[0]["title"][:80]
        except Exception as e:
            logger.debug(f"銘柄別ニュース取得失敗 ({code}): {e}")

        # 総合スコア（個別ニュース重視）
        total_score = company_score * 0.7 + all_news_score * 0.3
        total_score = max(NEWS_SCORE_MIN, min(NEWS_SCORE_MAX, total_score))

        # センチメントラベル
        if total_score > 3:
            label = "ポジティブ"
        elif total_score < -3:
            label = "ネガティブ"
        else:
            label = "中立"

        return total_score, label, company_summary

    def _score_news_dataframe(
        self,
        news_df: pd.DataFrame,
        keywords_filter: str | None = None,
    ) -> float:
        """ニュースDataFrameのセンチメントスコアを計算

        Args:
            news_df: ニュースデータ
            keywords_filter: フィルタキーワード（None=全件対象）

        Returns:
            -10.0 〜 +10.0 のスコア
        """
        if news_df.empty:
            return 0.0

        positive_count = 0
        negative_count = 0
        total_articles = 0

        for _, row in news_df.iterrows():
            text = str(row.get("title", "")) + " " + str(row.get("description", ""))

            if keywords_filter and keywords_filter not in text:
                continue

            total_articles += 1

            for kw in POSITIVE_KEYWORDS:
                if kw in text:
                    positive_count += 1

            for kw in NEGATIVE_KEYWORDS:
                if kw in text:
                    negative_count += 1

        if total_articles == 0:
            return 0.0

        # 正規化スコア
        net = positive_count - negative_count
        # 記事数で割って正規化し、±10にスケール
        normalized = net / max(total_articles, 1) * 5.0
        return max(NEWS_SCORE_MIN, min(NEWS_SCORE_MAX, normalized))

    def get_macro_score(self, market_data: MarketData) -> float:
        """マクロ環境スコアを取得"""
        return market_data.macro_indicators.get("macro_score", 0.0)
