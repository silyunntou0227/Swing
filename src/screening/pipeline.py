"""5層スクリーニングパイプラインオーケストレーター"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from src.config import (
    TARGET_MARKETS,
    EXCLUDE_CATEGORIES,
    MIN_LISTING_DAYS,
    ADX_MIN,
    ATR_RANGE_FILTER_ENABLED,
    ATR_RANGE_FILTER_MIN,
    ATR_RANGE_FILTER_MAX,
    VOL_RATIO_FILTER_ENABLED,
    SIGNAL_CONFLICT_FILTER_ENABLED,
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

# テクニカル分析の並列ワーカー数（環境変数で制御可能）
# 0 = CPUコア数自動検出、正の整数 = その数
_WORKERS_ENV = int(os.environ.get("ANALYSIS_WORKERS", "0"))
ANALYSIS_WORKERS = _WORKERS_ENV if _WORKERS_ENV > 0 else min(os.cpu_count() or 4, 8)


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


def _analyze_single_stock(args: tuple) -> dict | None:
    """単一銘柄のテクニカル分析（並列ワーカー用トップレベル関数）

    ProcessPoolExecutor はトップレベル関数のみピクル可能なため、
    クラスメソッドではなくモジュールレベルに配置する。
    args: (code, prices_values, config_dict)
      config_dict: {"adx_min", "atr_range_enabled", "atr_range_min",
                     "atr_range_max", "vol_ratio_enabled"}
    """
    code, prices_values, cfg = args
    adx_min = cfg.get("adx_min", ADX_MIN)
    atr_range_on = cfg.get("atr_range_enabled", ATR_RANGE_FILTER_ENABLED)
    atr_min = cfg.get("atr_range_min", ATR_RANGE_FILTER_MIN)
    atr_max = cfg.get("atr_range_max", ATR_RANGE_FILTER_MAX)
    vol_ratio_on = cfg.get("vol_ratio_enabled", VOL_RATIO_FILTER_ENABLED)

    try:
        stock_prices = pd.DataFrame(prices_values)
        if len(stock_prices) < 50:
            return None

        stock_prices = stock_prices.reset_index(drop=True)

        # テクニカル指標計算
        stock_prices = calculate_all_indicators(stock_prices)
        if len(stock_prices) < 30:
            return None

        last = stock_prices.iloc[-1]

        # ADXチェック
        if "ADX" in stock_prices.columns:
            adx_val = last.get("ADX", 0)
            if pd.notna(adx_val) and adx_val < adx_min:
                return None

        # レンジ相場フィルタ: ATR/Close比率でチョップゾーン・過剰ボラを除外
        if atr_range_on:
            close_val = float(last.get("Close", 0))
            atr_val = float(last.get("ATR", 0)) if "ATR" in stock_prices.columns else 0
            if close_val > 0 and atr_val > 0:
                atr_ratio = atr_val / close_val
                if atr_ratio < atr_min or atr_ratio > atr_max:
                    return None

        # 出来高フィルタ: 直近出来高が20日平均以上を必須化
        if vol_ratio_on:
            vol_ratio = float(last.get("VolumeRatio", 1.0)) if "VolumeRatio" in stock_prices.columns else 1.0
            if pd.notna(vol_ratio) and vol_ratio < 1.0:
                return None

        # シグナル検出
        signals = get_all_signals(stock_prices)
        if not signals:
            return None

        buy_count, sell_count = count_buy_sell_signals(signals)
        if buy_count == 0 and sell_count == 0:
            return None

        close = float(last.get("Close", 0))

        return {
            "code": code,
            "close": close,
            "signals": signals,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "prices_df": stock_prices,
        }
    except (KeyError, ValueError, TypeError):
        return None


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

        # 分析タスクを構築（データ不足の銘柄は事前除外）
        tasks = []
        skipped_short = 0
        for code in liquid_codes:
            prices_df = prices_by_code.get(code)
            if prices_df is None or len(prices_df) < 50:
                skipped_short += 1
                continue
            worker_cfg = {
                "adx_min": ADX_MIN,
                "atr_range_enabled": ATR_RANGE_FILTER_ENABLED,
                "atr_range_min": ATR_RANGE_FILTER_MIN,
                "atr_range_max": ATR_RANGE_FILTER_MAX,
                "vol_ratio_enabled": VOL_RATIO_FILTER_ENABLED,
            }
            tasks.append((code, prices_df.to_dict("list"), worker_cfg))

        total = len(tasks)
        logger.info(
            f"  テクニカル分析開始: {total}銘柄 "
            f"(ワーカー{ANALYSIS_WORKERS}並列, データ不足スキップ={skipped_short})"
        )

        # --- 並列テクニカル分析 ---
        candidates = []
        done_count = 0

        with ProcessPoolExecutor(max_workers=ANALYSIS_WORKERS) as executor:
            futures = {
                executor.submit(_analyze_single_stock, task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                done_count += 1
                if done_count % 200 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"  テクニカル分析中... {done_count}/{total} "
                        f"(候補{len(candidates)}件, {elapsed:.0f}s)"
                    )

                result = future.result()
                if result is None:
                    continue

                name = name_map.get(result["code"], "")
                candidates.append(CandidateStock(
                    code=result["code"],
                    name=name or result["code"],
                    close=result["close"],
                    signals=result["signals"],
                    buy_signal_count=result["buy_count"],
                    sell_signal_count=result["sell_count"],
                    prices_df=result["prices_df"],
                ))

        logger.info(
            f"  統計: データ不足={skipped_short}, "
            f"分析済={total}, 候補={len(candidates)}"
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

            buy_c = candidate.buy_signal_count
            sell_c = candidate.sell_signal_count
            if SIGNAL_CONFLICT_FILTER_ENABLED:
                # 信号矛盾フィルタ: 優勢側が逆側の2倍以上 & 最低2シグナル
                if buy_c >= 2 and buy_c >= sell_c * 2:
                    result.buy.append(candidate)
                elif sell_c >= 2 and sell_c >= buy_c * 2:
                    result.sell.append(candidate)
            else:
                # 単純多数決
                if buy_c > sell_c:
                    result.buy.append(candidate)
                elif sell_c > buy_c:
                    result.sell.append(candidate)

        return result
