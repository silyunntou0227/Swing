"""データ取得・統合・前処理モジュール"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from src.data.jquants_client import JQuantsClient
from src.data.tdnet_client import TDnetClient
from src.data.edinet_client import EDINETClient
from src.data.yahoo_client import YahooClient
from src.data.news_client import NewsClient
from src.data.macro_client import MacroClient
from src.data.margin_client import MarginClient
from src.utils.logging_config import logger


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
    """全データソースからデータを取得・統合するローダー"""

    def __init__(self) -> None:
        self._jquants = JQuantsClient()

    def load_all(self) -> MarketData:
        """全データソースからデータを取得"""
        data = MarketData()

        # === 必須データ（J-Quants） ===
        try:
            data.stocks = self._jquants.fetch_listed_stocks()
        except Exception as e:
            logger.warning(f"上場銘柄一覧取得失敗（続行）: {e}")
        data.prices = self._load_prices()
        data.financials = self._load_financials()

        # === 補助データ（失敗しても続行） ===
        data.disclosures = self._load_disclosures()
        data.edinet_filings = self._load_edinet()
        data.news = self._load_news()
        data.margin_data = self._load_margin()
        data.macro_indicators = self._load_macro()

        return data

    def _load_prices(self) -> pd.DataFrame:
        """株価データ取得・前処理"""
        df = self._jquants.fetch_daily_quotes()
        if df.empty:
            return df

        # 調整済み株価を使用
        rename_map = {
            "AdjustmentOpen": "Open",
            "AdjustmentHigh": "High",
            "AdjustmentLow": "Low",
            "AdjustmentClose": "Close",
            "AdjustmentVolume": "Volume",
        }
        # 調整済みカラムが存在する場合のみリネーム
        existing_adj = {k: v for k, v in rename_map.items() if k in df.columns}
        if existing_adj:
            # 元のカラムを削除してからリネーム
            for new_name in existing_adj.values():
                if new_name in df.columns:
                    df = df.drop(columns=[new_name])
            df = df.rename(columns=existing_adj)

        # 日付型変換
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        # 銘柄コード正規化（4桁に）
        if "Code" in df.columns:
            df["Code"] = df["Code"].astype(str).str[:4]

        # 欠損値のある行を除外
        price_cols = ["Open", "High", "Low", "Close", "Volume"]
        existing_price_cols = [c for c in price_cols if c in df.columns]
        df = df.dropna(subset=existing_price_cols)

        # ソート
        df = df.sort_values(["Code", "Date"]).reset_index(drop=True)

        logger.info(f"株価データ前処理完了: {df['Code'].nunique()}銘柄")
        return df

    def _load_financials(self) -> pd.DataFrame:
        """財務データ取得・前処理"""
        df = self._jquants.fetch_financial_data()
        if df.empty:
            return df

        # 銘柄コード正規化
        code_col = "LocalCode" if "LocalCode" in df.columns else "Code"
        if code_col in df.columns:
            df["Code"] = df[code_col].astype(str).str[:4]

        # 最新の財務データのみ残す（銘柄ごと）
        if "DisclosedDate" in df.columns:
            df["DisclosedDate"] = pd.to_datetime(df["DisclosedDate"])
            df = df.sort_values("DisclosedDate").groupby("Code").tail(1)

        logger.info(f"財務データ前処理完了: {len(df)}銘柄")
        return df

    def _load_disclosures(self) -> pd.DataFrame:
        """TDnet適時開示データ取得"""
        try:
            client = TDnetClient()
            df = client.fetch_today_disclosures()
            logger.info(f"TDnet適時開示: {len(df)}件取得")
            return df
        except Exception as e:
            logger.warning(f"TDnet適時開示取得失敗（続行）: {e}")
            return pd.DataFrame()

    def _load_edinet(self) -> pd.DataFrame:
        """EDINET開示データ取得"""
        try:
            client = EDINETClient()
            df = client.fetch_recent_filings()
            logger.info(f"EDINET開示: {len(df)}件取得")
            return df
        except Exception as e:
            logger.warning(f"EDINET開示取得失敗（続行）: {e}")
            return pd.DataFrame()

    def _load_news(self) -> pd.DataFrame:
        """ニュースデータ取得"""
        try:
            client = NewsClient()
            df = client.fetch_market_news()
            logger.info(f"ニュース: {len(df)}件取得")
            return df
        except Exception as e:
            logger.warning(f"ニュース取得失敗（続行）: {e}")
            return pd.DataFrame()

    def _load_margin(self) -> pd.DataFrame:
        """信用残・空売りデータ取得"""
        try:
            client = MarginClient()
            df = client.fetch_margin_data()
            logger.info(f"信用残データ: {len(df)}件取得")
            return df
        except Exception as e:
            logger.warning(f"信用残データ取得失敗（続行）: {e}")
            return pd.DataFrame()

    def _load_macro(self) -> dict:
        """マクロ指標取得"""
        try:
            client = MacroClient()
            indicators = client.fetch_indicators()
            logger.info(f"マクロ指標: {len(indicators)}項目取得")
            return indicators
        except Exception as e:
            logger.warning(f"マクロ指標取得失敗（続行）: {e}")
            return {}
