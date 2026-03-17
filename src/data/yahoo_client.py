"""Yahoo Finance Japan 補助データ取得クライアント"""

from __future__ import annotations

import pandas as pd

from src.utils.logging_config import logger


class YahooClient:
    """Yahoo Finance Japan からの補助データ取得

    yfinance ライブラリを使用して信用残・補完データを取得する。
    J-Quantsが主データソースのため、こちらは補完用途。
    """

    def __init__(self) -> None:
        try:
            import yfinance
            self._yf = yfinance
        except ImportError:
            logger.warning("yfinance がインストールされていません")
            self._yf = None

    def fetch_margin_info(self, codes: list[str]) -> pd.DataFrame:
        """信用残情報を取得（Yahoo Finance経由）

        Args:
            codes: 銘柄コードリスト（4桁）

        Returns:
            columns: code, margin_buy, margin_sell, margin_ratio
        """
        if self._yf is None:
            return pd.DataFrame()

        records = []
        for code in codes:
            try:
                ticker = self._yf.Ticker(f"{code}.T")
                info = ticker.info or {}

                # Yahoo Finance の info から取得可能なフィールド
                records.append({
                    "code": code,
                    "market_cap": info.get("marketCap", None),
                    "trailing_pe": info.get("trailingPE", None),
                    "forward_pe": info.get("forwardPE", None),
                    "price_to_book": info.get("priceToBook", None),
                    "dividend_yield": info.get("dividendYield", None),
                    "fifty_day_average": info.get("fiftyDayAverage", None),
                    "two_hundred_day_average": info.get("twoHundredDayAverage", None),
                })
            except Exception as e:
                logger.debug(f"Yahoo Finance {code} データ取得失敗: {e}")
                continue

        return pd.DataFrame(records) if records else pd.DataFrame()

    def fetch_stock_price(
        self, code: str, period: str = "1y"
    ) -> pd.DataFrame:
        """単一銘柄の株価データを取得（J-Quants補完用）

        Args:
            code: 銘柄コード（4桁）
            period: 取得期間（"1y", "6mo", etc.）
        """
        if self._yf is None:
            return pd.DataFrame()

        try:
            ticker = self._yf.Ticker(f"{code}.T")
            df = ticker.history(period=period)
            if not df.empty:
                df = df.reset_index()
                df["Code"] = code
            return df
        except Exception as e:
            logger.debug(f"Yahoo Finance {code} 株価取得失敗: {e}")
            return pd.DataFrame()
