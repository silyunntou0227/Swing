"""Yahoo Finance Japan データ取得クライアント（主力データソース）

J-Quants 無料プランの厳しい制限を回避するため、
yfinance を株価データの主力取得手段として使用する。
yfinance はレート制限が緩く、一括ダウンロードにも対応している。
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

from src.config import HISTORY_DAYS
from src.utils.logging_config import logger


class YahooClient:
    """Yahoo Finance から株価・基本情報を取得するクライアント

    主な機能:
    1. 全銘柄の株価データ一括取得（yfinance.download）
    2. 個別銘柄の基本情報取得（PER, PBR等）
    3. 信用残情報の補完取得
    """

    def __init__(self) -> None:
        try:
            import yfinance
            self._yf = yfinance
        except ImportError:
            logger.error("yfinance がインストールされていません — pip install yfinance")
            self._yf = None

    def fetch_bulk_prices(
        self,
        codes: list[str],
        period: str = "1y",
    ) -> pd.DataFrame:
        """複数銘柄の株価データを一括取得

        yfinance.download() を使用して効率的にバルクダウンロード。
        GitHub Actions 環境でも安定して動作する。

        Args:
            codes: 銘柄コードリスト（4桁）
            period: 取得期間（"1y"=1年, "6mo"=6ヶ月等）

        Returns:
            統一フォーマットのDataFrame（Code, Date, Open, High, Low, Close, Volume）
        """
        if self._yf is None:
            return pd.DataFrame()

        if not codes:
            logger.warning("Yahoo: 銘柄リストが空です")
            return pd.DataFrame()

        # 銘柄コードを Yahoo Finance 形式に変換（4桁 → 4桁.T）
        tickers = [f"{code}.T" for code in codes]
        logger.info(f"Yahoo Finance: {len(tickers)}銘柄の株価を一括取得中 (period={period})...")

        all_frames = []

        # yfinance.download はバッチで処理可能だが、
        # 大量（4000銘柄）の場合はチャンクに分割して安定性を確保
        chunk_size = 200
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(tickers) + chunk_size - 1) // chunk_size
            logger.info(f"Yahoo Finance: チャンク {chunk_num}/{total_chunks} ({len(chunk)}銘柄)")

            try:
                df = self._yf.download(
                    tickers=chunk,
                    period=period,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                )

                if df.empty:
                    continue

                # マルチカラムインデックスを展開してフラット化
                chunk_df = self._flatten_multi_ticker_df(df, chunk)
                if not chunk_df.empty:
                    all_frames.append(chunk_df)

            except Exception as e:
                logger.warning(f"Yahoo Finance: チャンク {chunk_num} 取得エラー: {e}")
                # チャンクが失敗した場合、個別取得にフォールバック
                for ticker in chunk:
                    try:
                        single_df = self._fetch_single_ticker(ticker, period)
                        if not single_df.empty:
                            all_frames.append(single_df)
                    except Exception:
                        pass

            # チャンク間の小休止
            if i + chunk_size < len(tickers):
                time.sleep(1)

        if not all_frames:
            logger.warning("Yahoo Finance: 株価データ取得件数0")
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        logger.info(
            f"Yahoo Finance: 株価データ取得完了 — "
            f"{result['Code'].nunique()}銘柄, {len(result)}行"
        )
        return result

    def _flatten_multi_ticker_df(
        self, df: pd.DataFrame, tickers: list[str]
    ) -> pd.DataFrame:
        """yfinance.download のマルチティッカー結果をフラット化

        yfinance.download(group_by='ticker') の結果は
        マルチカラムインデックス [(ticker, OHLCV)] になる。
        これを Code, Date, Open, ... の形式に変換する。
        """
        frames = []

        # 1銘柄のみの場合はマルチインデックスにならない
        if len(tickers) == 1:
            ticker = tickers[0]
            code = ticker.replace(".T", "")
            temp = df.copy().reset_index()
            if "Date" not in temp.columns and "index" in temp.columns:
                temp = temp.rename(columns={"index": "Date"})
            temp["Code"] = code
            # カラム名を統一
            temp = self._normalize_columns(temp)
            if not temp.empty:
                frames.append(temp)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        for ticker in tickers:
            code = ticker.replace(".T", "")
            try:
                if ticker in df.columns.get_level_values(0):
                    sub = df[ticker].copy()
                elif ticker.upper() in df.columns.get_level_values(0):
                    sub = df[ticker.upper()].copy()
                else:
                    continue

                sub = sub.reset_index()
                if "Date" not in sub.columns and "index" in sub.columns:
                    sub = sub.rename(columns={"index": "Date"})

                sub["Code"] = code
                sub = self._normalize_columns(sub)

                # NaN行を除去
                if "Close" in sub.columns:
                    sub = sub.dropna(subset=["Close"])

                if not sub.empty:
                    frames.append(sub)
            except Exception:
                continue

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_single_ticker(self, ticker: str, period: str) -> pd.DataFrame:
        """単一銘柄の株価を取得（フォールバック用）"""
        if self._yf is None:
            return pd.DataFrame()

        code = ticker.replace(".T", "")
        try:
            obj = self._yf.Ticker(ticker)
            df = obj.history(period=period)
            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            df["Code"] = code
            df = self._normalize_columns(df)
            return df
        except Exception as e:
            logger.debug(f"Yahoo Finance: {code} 個別取得失敗: {e}")
            return pd.DataFrame()

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """カラム名を統一フォーマットに正規化"""
        rename_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower == "open":
                rename_map[col] = "Open"
            elif col_lower == "high":
                rename_map[col] = "High"
            elif col_lower == "low":
                rename_map[col] = "Low"
            elif col_lower == "close":
                rename_map[col] = "Close"
            elif col_lower == "volume":
                rename_map[col] = "Volume"
            elif col_lower == "date":
                rename_map[col] = "Date"

        if rename_map:
            df = df.rename(columns=rename_map)

        # 必要な列のみ抽出
        required_cols = ["Code", "Date", "Open", "High", "Low", "Close", "Volume"]
        existing = [c for c in required_cols if c in df.columns]
        if len(existing) >= 3:  # 最低限 Code, Date, Close
            df = df[existing]

        return df

    def fetch_stock_price(
        self, code: str, period: str = "1y"
    ) -> pd.DataFrame:
        """単一銘柄の株価データを取得

        Args:
            code: 銘柄コード（4桁）
            period: 取得期間（"1y", "6mo", etc.）
        """
        return self._fetch_single_ticker(f"{code}.T", period)

    def fetch_basic_info(self, codes: list[str]) -> pd.DataFrame:
        """銘柄の基本情報（PER, PBR, ROE等）を yfinance から取得

        J-Quants 無料プランで財務データ取得不可（403）の代替。

        Args:
            codes: 銘柄コードリスト（4桁）

        Returns:
            columns: Code, market_cap, trailing_pe, forward_pe,
                     price_to_book, dividend_yield, roe, company_name
        """
        if self._yf is None:
            return pd.DataFrame()

        records = []
        total = len(codes)
        for i, code in enumerate(codes):
            if i > 0 and i % 50 == 0:
                logger.info(f"Yahoo Finance: 基本情報取得中... {i}/{total}")
                time.sleep(1)

            try:
                ticker = self._yf.Ticker(f"{code}.T")
                info = ticker.info or {}

                records.append({
                    "Code": code,
                    "CompanyName": info.get("longName") or info.get("shortName", code),
                    "market_cap": info.get("marketCap"),
                    "trailing_pe": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "price_to_book": info.get("priceToBook"),
                    "dividend_yield": info.get("dividendYield"),
                    "roe": info.get("returnOnEquity"),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                })
            except Exception as e:
                logger.debug(f"Yahoo Finance: {code} 基本情報取得失敗: {e}")
                continue

        df = pd.DataFrame(records) if records else pd.DataFrame()
        if not df.empty:
            logger.info(f"Yahoo Finance: 基本情報 {len(df)}銘柄取得")
        return df

    def fetch_margin_info(self, codes: list[str]) -> pd.DataFrame:
        """信用残情報を取得（Yahoo Finance経由）

        Args:
            codes: 銘柄コードリスト（4桁）

        Returns:
            columns: code, market_cap, trailing_pe, forward_pe,
                     price_to_book, dividend_yield
        """
        if self._yf is None:
            return pd.DataFrame()

        records = []
        for code in codes:
            try:
                ticker = self._yf.Ticker(f"{code}.T")
                info = ticker.info or {}

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
