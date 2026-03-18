"""データ取得・統合・前処理モジュール

データ取得戦略（2026年3月時点）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
| データ種別      | 主力ソース         | フォールバック     |
|-----------------|--------------------|--------------------|
| 銘柄一覧        | JPX公式CSV         | J-Quants V2       |
| 株価OHLCV       | yfinance           | J-Quants V2       |
| 財務データ      | yfinance (info)    | J-Quants V2       |
| 適時開示        | TDnet              | —                  |
| EDINET          | EDINET API         | —                  |
| ニュース        | NewsAPI/Google RSS | —                  |
| 信用残          | Yahoo/JPXスクレイピング | —             |
| マクロ指標      | yfinance (^N225)   | —                  |
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import requests

from src.data.stock_list import fetch_jpx_stock_list, get_tradeable_codes
from src.data.yahoo_client import YahooClient
from src.config import JQUANTS_API_KEY
from src.utils.logging_config import logger

# GitHub Actions 無料枠（2コア, 60分制限）で安全に完了するための銘柄上限
# プライム全銘柄(~1800) でスイングトレードの主要対象をカバー
# ローカル実行では MAX_STOCKS_FOR_DOWNLOAD=0 で全銘柄対応可
MAX_STOCKS_FOR_DOWNLOAD = 1200


@dataclass
class MarketData:
    """全データソースの統合データ"""

    stocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    financials: pd.DataFrame = field(default_factory=pd.DataFrame)
    disclosures: pd.DataFrame = field(default_factory=pd.DataFrame)
    edinet_filings: pd.DataFrame = field(default_factory=pd.DataFrame)
    news: pd.DataFrame = field(default_factory=pd.DataFrame)
    margin_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    macro_indicators: dict = field(default_factory=dict)
    scan_date: date = field(default_factory=date.today)

    @property
    def has_prices(self) -> bool:
        return not self.prices.empty

    @property
    def has_financials(self) -> bool:
        return not self.financials.empty

    @property
    def has_news(self) -> bool:
        return not self.news.empty

    @property
    def has_disclosures(self) -> bool:
        return not self.disclosures.empty

    @property
    def has_margin(self) -> bool:
        return not self.margin_data.empty


class DataLoader:
    """全データソースからデータを取得・統合するローダー

    yfinance を主力とし、J-Quants は補助的に使用する。
    全てのデータ取得は非致命的（失敗しても続行）。
    """

    def __init__(self) -> None:
        self._yahoo = YahooClient()
        self._jquants = None

        # J-Quants は API キーがある場合のみ初期化
        if JQUANTS_API_KEY:
            try:
                from src.data.jquants_client import JQuantsClient
                self._jquants = JQuantsClient()
            except Exception as e:
                logger.warning(f"J-Quants クライアント初期化失敗（続行）: {e}")

    def load_all(self) -> MarketData:
        """全データソースからデータを取得"""
        data = MarketData()
        t0 = time.time()

        # === Step 1: 銘柄一覧取得（JPX公式 → J-Quants フォールバック）===
        logger.info("=" * 50)
        logger.info("Step 1/7: 銘柄一覧取得")
        data.stocks = self._load_stock_list()
        codes = get_tradeable_codes(data.stocks, max_stocks=MAX_STOCKS_FOR_DOWNLOAD)
        logger.info(f"分析対象: {len(codes)}銘柄（{time.time()-t0:.0f}秒経過）")

        # === Step 2: 株価データ取得（yfinance 主力）===
        logger.info("=" * 50)
        logger.info("Step 2/7: 株価データ取得（yfinance）")
        t1 = time.time()
        data.prices = self._load_prices(codes)
        logger.info(f"株価取得完了（{time.time()-t1:.0f}秒）")

        # === Step 3: 財務データ（J-Quants → スキップ）===
        logger.info("=" * 50)
        logger.info("Step 3/7: 財務データ取得")
        data.financials = self._load_financials()

        # === Step 4-7: 補助データ（全て非致命的）===
        logger.info("=" * 50)
        logger.info("Step 4-7: 補助データ取得")
        data.disclosures = self._load_disclosures()
        data.edinet_filings = self._load_edinet()
        data.news = self._load_news()
        data.macro_indicators = self._load_macro()
        # 信用残は後でスクリーニング候補に対して個別取得
        data.margin_data = pd.DataFrame()

        elapsed = time.time() - t0
        logger.info(f"全データ取得完了（{elapsed:.0f}秒）")

        return data

    def _load_stock_list(self) -> pd.DataFrame:
        """銘柄一覧取得（JPX公式CSV → J-Quantsフォールバック）"""
        # 主力: JPX公式CSV
        df = fetch_jpx_stock_list()
        if not df.empty:
            return df

        # フォールバック: J-Quants V2
        if self._jquants:
            try:
                df = self._jquants.fetch_listed_stocks()
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"J-Quants 銘柄一覧取得失敗: {e}")

        logger.error("銘柄一覧を取得できませんでした（全ソース失敗）")
        return pd.DataFrame()

    def _load_prices(self, codes: list[str]) -> pd.DataFrame:
        """株価データ取得・前処理（yfinance主力）"""
        if not codes:
            logger.warning("銘柄コードリストが空 — 株価データ取得スキップ")
            return pd.DataFrame()

        # yfinance で一括取得（レート制限なし、高速）
        df = self._yahoo.fetch_bulk_prices(codes, period="1y")

        if df.empty:
            logger.warning("yfinance からの株価データ取得失敗")
            return pd.DataFrame()

        # 前処理
        df = self._preprocess_prices(df)
        return df

    def _preprocess_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """株価データの前処理"""
        if df.empty:
            return df

        # 日付型変換（timezone-aware → naive に統一）
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)

        # 銘柄コード正規化（4桁に）
        if "Code" in df.columns:
            df["Code"] = df["Code"].astype(str).str[:4]

        # 欠損値のある行を除外
        price_cols = ["Open", "High", "Low", "Close", "Volume"]
        existing_price_cols = [c for c in price_cols if c in df.columns]
        if existing_price_cols:
            df = df.dropna(subset=existing_price_cols)

        # ソート
        if "Code" in df.columns and "Date" in df.columns:
            df = df.sort_values(["Code", "Date"]).reset_index(drop=True)

        logger.info(f"株価データ前処理完了: {df['Code'].nunique()}銘柄, {len(df)}行")
        return df

    def _load_financials(self) -> pd.DataFrame:
        """財務データ取得（J-Quants → 失敗時は空で続行）

        J-Quants 無料プランでは /fins/statements が 403 のため、
        取得失敗時はファンダメンタルフィルタをスキップする設計。
        """
        if self._jquants:
            try:
                df = self._jquants.fetch_financial_data()
                if not df.empty:
                    # 銘柄コード正規化
                    code_col = "LocalCode" if "LocalCode" in df.columns else "Code"
                    if code_col in df.columns:
                        df["Code"] = df[code_col].astype(str).str[:4]
                    if "DisclosedDate" in df.columns:
                        df["DisclosedDate"] = pd.to_datetime(df["DisclosedDate"])
                        df = df.sort_values("DisclosedDate").groupby("Code").tail(1)
                    logger.info(f"財務データ前処理完了: {len(df)}銘柄")
                    return df
            except Exception as e:
                logger.warning(f"財務データ取得失敗（続行）: {e}")

        logger.info("財務データなし — ファンダメンタルフィルタはバイパスされます")
        return pd.DataFrame()

    def _load_disclosures(self) -> pd.DataFrame:
        """TDnet適時開示データ取得"""
        try:
            from src.data.tdnet_client import TDnetClient
            client = TDnetClient()
            df = client.fetch_today_disclosures()
            logger.info(f"TDnet適時開示: {len(df)}件取得")
            return df
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"TDnet適時開示: ネットワークエラー（続行）: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"TDnet適時開示取得失敗（続行）: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def _load_edinet(self) -> pd.DataFrame:
        """EDINET開示データ取得"""
        try:
            from src.data.edinet_client import EDINETClient
            client = EDINETClient()
            df = client.fetch_recent_filings()
            logger.info(f"EDINET開示: {len(df)}件取得")
            return df
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"EDINET: ネットワークエラー（続行）: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"EDINET開示取得失敗（続行）: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def _load_news(self) -> pd.DataFrame:
        """ニュースデータ取得"""
        try:
            from src.data.news_client import NewsClient
            client = NewsClient()
            df = client.fetch_market_news()
            logger.info(f"ニュース: {len(df)}件取得")
            return df
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"ニュース: ネットワークエラー（続行）: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"ニュース取得失敗（続行）: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def _load_macro(self) -> dict:
        """マクロ指標取得"""
        try:
            from src.data.macro_client import MacroClient
            client = MacroClient()
            indicators = client.fetch_indicators()
            logger.info(f"マクロ指標: {len(indicators)}項目取得")
            return indicators
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"マクロ指標: ネットワークエラー（続行）: {e}")
            return {}
        except Exception as e:
            logger.warning(f"マクロ指標取得失敗（続行）: {type(e).__name__}: {e}")
            return {}
