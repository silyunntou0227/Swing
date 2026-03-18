"""5層スクリーニングパイプラインオーケストレーター"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from src.config import (
    TARGET_MARKETS,
    EXCLUDE_CATEGORIES,
    MIN_LISTING_DAYS,
    ADX_MIN,
    SIGNAL_LOOKBACK_DAYS,
)
from src.data.data_loader import MarketData
from src.screening.fundamental import filter_fundamentals
from src.screening.liquidity import filter_liquidity
from src.screening.news_filter import NewsFilter
from src.indicators.technical import (
    calculate_all_indicators,
    get_all_signals,
    count_buy_sell_signals,
)
from src.utils.logging_config import logger


@dataclass
class CandidateStock:
    """スクリーニング通過した候補銘柄"""

    code: str
    name: str
    close: float
    signals: list[str] = field(default_factory=list)
    buy_signal_count: int = 0
    sell_signal_count: int = 0
    prices_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ScreeningResult:
    """スクリーニング結果"""

    buy: list[CandidateStock] = field(default_factory=list)
    sell: list[CandidateStock] = field(default_factory=list)


class ScreeningPipeline:
    """5層スクリーニングパイプライン

    Layer 0: ユニバースフィルタ（市場・銘柄種別）
    Layer 1: ファンダメンタルフィルタ
    Layer 2+3: トレンド整合性・流動性 + エントリーシグナル（統合処理）
    Layer 4: ニュース・開示フィルタ

    最適化: Layer 2 と Layer 3 でテクニカル指標計算を1回に統合
    """

    def __init__(self) -> None:
        self._news_filter = NewsFilter()

    def run(self, market_data: MarketData) -> ScreeningResult:
        """5層パイプラインを実行"""
        result = ScreeningResult()
        t0 = time.time()

        # Layer 0: ユニバースフィルタ
        universe = self._layer0_universe(market_data)
        logger.info(f"Layer 0 (ユニバース): {len(universe)}銘柄 [{time.time()-t0:.0f}s]")

        # Layer 1: ファンダメンタル
        filtered = self._layer1_fundamental(universe, market_data)
        logger.info(f"Layer 1 (ファンダメンタル): {len(filtered)}銘柄 [{time.time()-t0:.0f}s]")

        # Layer 2+3: トレンド・流動性 → エントリーシグナル（統合処理）
        codes = filtered["Code"].tolist() if "Code" in filtered.columns else []
        candidates = self._layer2_3_combined(codes, market_data, filtered)
        buy_count = len([c for c in candidates if c.buy_signal_count > c.sell_signal_count])
        sell_count = len([c for c in candidates if c.sell_signal_count > c.buy_signal_count])
        logger.info(
            f"Layer 2+3 (トレンド+シグナル): "
            f"買い{buy_count}件, 売り{sell_count}件 [{time.time()-t0:.0f}s]"
        )

        # Layer 4: ニュース・開示フィルタ
        result = self._layer4_news_filter(candidates, market_data)
        logger.info(
            f"Layer 4 (ニュース・開示): 買い{len(result.buy)}件, 売り{len(result.sell)}件 "
            f"[{time.time()-t0:.0f}s]"
        )

        return result

    def _layer0_universe(self, market_data: MarketData) -> pd.DataFrame:
        """Layer 0: ユニバースフィルタ"""
        stocks = market_data.stocks.copy()
        if stocks.empty:
            logger.warning("Layer 0: 銘柄リストが空です — データ取得に失敗した可能性があります")
            return stocks

        # 対象市場フィルタ
        if "MarketCodeName" in stocks.columns:
            stocks = stocks[
                stocks["MarketCodeName"].isin(TARGET_MARKETS)
                | stocks["MarketCodeName"].str.contains(
                    "|".join(TARGET_MARKETS), na=False
                )
            ]

        # ETF/REIT除外
        if "Sector17CodeName" in stocks.columns:
            for cat in EXCLUDE_CATEGORIES:
                stocks = stocks[~stocks["Sector17CodeName"].str.contains(cat, na=False)]

        # 銘柄コード正規化
        if "Code" in stocks.columns:
            stocks["Code"] = stocks["Code"].astype(str).str[:4]

        return stocks

    def _layer1_fundamental(
        self, stocks: pd.DataFrame, market_data: MarketData
    ) -> pd.DataFrame:
        """Layer 1: ファンダメンタルフィルタ"""
        return filter_fundamentals(stocks, market_data.financials)

    def _layer2_3_combined(
        self,
        codes: list[str],
        market_data: MarketData,
        stocks_info: pd.DataFrame,
    ) -> list[CandidateStock]:
        """Layer 2+3 統合: トレンド・流動性チェック + エントリーシグナル検出

        従来は Layer 2 と Layer 3 で calculate_all_indicators を2回呼んでいたが、
        1回の計算で両方のチェックを行うことで処理時間を半減する。
        """
        t0 = time.time()

        # まず流動性でフィルタ
        liquid_codes = filter_liquidity(codes, market_data.prices)

        # 価格データを銘柄ごとにグループ化（ループ内での繰り返しフィルタを回避）
        liquid_set = set(liquid_codes)
        prices_by_code = {
            code: group
            for code, group in market_data.prices[
                market_data.prices["Code"].isin(liquid_set)
            ].groupby("Code")
        }

        # 銘柄名辞書を事前構築（ループ内のN+1フィルタリングを回避）
        name_map: dict[str, str] = {}
        if not stocks_info.empty and "Code" in stocks_info.columns:
            for _, row in stocks_info[["Code", "CompanyName"]].dropna().iterrows():
                name_map[row["Code"]] = row.get("CompanyName", "")

        candidates = []
        total = len(liquid_codes)
        skipped_short = 0
        skipped_adx = 0
        skipped_signal = 0

        for i, code in enumerate(liquid_codes):
            # 進捗ログ（200銘柄ごと）
            if i > 0 and i % 200 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"  テクニカル分析中... {i}/{total} "
                    f"(候補{len(candidates)}件, {elapsed:.0f}s)"
                )

            stock_prices = prices_by_code.get(code)
            if stock_prices is None or len(stock_prices) < 50:
                skipped_short += 1
                continue

            stock_prices = stock_prices.reset_index(drop=True)

            # テクニカル指標を1回だけ計算（Layer 2+3 共有）
            try:
                stock_prices = calculate_all_indicators(stock_prices)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"テクニカル指標計算失敗 ({code}): {e}")
                continue

            if len(stock_prices) < 30:
                skipped_short += 1
                continue

            last = stock_prices.iloc[-1]

            # --- Layer 2 チェック: ADX（トレンド存在確認）---
            if "ADX" in stock_prices.columns:
                adx_val = last.get("ADX", 0)
                if pd.notna(adx_val) and adx_val < ADX_MIN:
                    skipped_adx += 1
                    continue

            # --- Layer 3 チェック: エントリーシグナル ---
            try:
                signals = get_all_signals(stock_prices)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"シグナル検出失敗 ({code}): {e}")
                continue

            if not signals:
                skipped_signal += 1
                continue

            buy_count, sell_count = count_buy_sell_signals(signals)

            # 少なくとも1つの買い/売りシグナルが必要
            if buy_count == 0 and sell_count == 0:
                skipped_signal += 1
                continue

            # 銘柄名取得（事前構築した辞書から O(1) で取得）
            name = name_map.get(code, "")

            close = stock_prices["Close"].iloc[-1] if "Close" in stock_prices.columns else 0.0

            candidates.append(CandidateStock(
                code=code,
                name=name or code,
                close=close,
                signals=signals,
                buy_signal_count=buy_count,
                sell_signal_count=sell_count,
                prices_df=stock_prices,
            ))

        logger.info(
            f"  統計: データ不足={skipped_short}, ADXフィルタ={skipped_adx}, "
            f"シグナルなし={skipped_signal}, 候補={len(candidates)}"
        )

        return candidates

    def _layer4_news_filter(
        self,
        candidates: list[CandidateStock],
        market_data: MarketData,
    ) -> ScreeningResult:
        """Layer 4: ニュース・開示フィルタ"""
        result = ScreeningResult()

        for candidate in candidates:
            # MBO/TOB等の除外チェック
            if self._news_filter.should_exclude(candidate.code, market_data):
                continue

            # 買い/売り判定
            if candidate.buy_signal_count > candidate.sell_signal_count:
                result.buy.append(candidate)
            elif candidate.sell_signal_count > candidate.buy_signal_count:
                result.sell.append(candidate)

        return result
