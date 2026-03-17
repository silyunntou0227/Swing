"""経済ニュースAPI取得クライアント"""

from __future__ import annotations

from datetime import date, timedelta

import feedparser
import pandas as pd
import requests

from src.config import NEWS_API_KEY
from src.utils.logging_config import logger

# Google News RSS（日本語・ビジネスカテゴリ）
GOOGLE_NEWS_RSS = "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtcGhHZ0pLVUNnQVAB?hl=ja&gl=JP&ceid=JP:ja"

# NewsAPI
NEWSAPI_URL = "https://newsapi.org/v2/everything"


class NewsClient:
    """経済ニュース取得クライアント

    優先順位:
    1. NewsAPI.org（APIキーがある場合）
    2. Google News RSS（フォールバック、常に無料）
    """

    def __init__(self) -> None:
        self._api_key = NEWS_API_KEY

    def fetch_market_news(self, days_back: int = 1) -> pd.DataFrame:
        """市場関連ニュースを取得

        Returns:
            columns: title, description, source, published_at, url
        """
        if self._api_key:
            df = self._fetch_from_newsapi(days_back)
            if not df.empty:
                return df

        # フォールバック: Google News RSS
        return self._fetch_from_google_news()

    def fetch_news_for_keyword(self, keyword: str) -> pd.DataFrame:
        """特定キーワードのニュースを検索

        Args:
            keyword: 検索キーワード（銘柄名など）
        """
        if self._api_key:
            return self._fetch_from_newsapi_keyword(keyword)
        return self._fetch_from_google_news_keyword(keyword)

    def _fetch_from_newsapi(self, days_back: int) -> pd.DataFrame:
        """NewsAPI.org から日本市場ニュースを取得"""
        from_date = (date.today() - timedelta(days=days_back)).isoformat()

        params = {
            "q": "株式 OR 日経平均 OR 東証 OR 決算",
            "language": "ja",
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": self._api_key,
        }

        try:
            resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"NewsAPI エラー: {e}")
            return pd.DataFrame()

        articles = data.get("articles", [])
        records = [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
                "url": a.get("url", ""),
            }
            for a in articles
        ]
        return pd.DataFrame(records)

    def _fetch_from_newsapi_keyword(self, keyword: str) -> pd.DataFrame:
        """NewsAPI.org からキーワード検索"""
        params = {
            "q": keyword,
            "language": "ja",
            "sortBy": "relevancy",
            "pageSize": 10,
            "apiKey": self._api_key,
        }

        try:
            resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"NewsAPI キーワード検索エラー: {e}")
            return pd.DataFrame()

        articles = data.get("articles", [])
        records = [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", ""),
                "published_at": a.get("publishedAt", ""),
                "url": a.get("url", ""),
            }
            for a in articles
        ]
        return pd.DataFrame(records)

    def _fetch_from_google_news(self) -> pd.DataFrame:
        """Google News RSS から日本のビジネスニュースを取得"""
        try:
            feed = feedparser.parse(GOOGLE_NEWS_RSS)
        except Exception as e:
            logger.warning(f"Google News RSS エラー: {e}")
            return pd.DataFrame()

        records = [
            {
                "title": entry.get("title", ""),
                "description": entry.get("summary", ""),
                "source": entry.get("source", {}).get("title", "")
                if isinstance(entry.get("source"), dict)
                else "",
                "published_at": entry.get("published", ""),
                "url": entry.get("link", ""),
            }
            for entry in feed.entries[:50]
        ]
        return pd.DataFrame(records)

    def _fetch_from_google_news_keyword(self, keyword: str) -> pd.DataFrame:
        """Google News RSS でキーワード検索"""
        import urllib.parse

        encoded = urllib.parse.quote(keyword)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=ja&gl=JP&ceid=JP:ja"

        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.warning(f"Google News キーワード検索エラー: {e}")
            return pd.DataFrame()

        records = [
            {
                "title": entry.get("title", ""),
                "description": entry.get("summary", ""),
                "source": entry.get("source", {}).get("title", "")
                if isinstance(entry.get("source"), dict)
                else "",
                "published_at": entry.get("published", ""),
                "url": entry.get("link", ""),
            }
            for entry in feed.entries[:10]
        ]
        return pd.DataFrame(records)
