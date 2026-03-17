"""加重マルチファクタースコアリングモジュール (0-100)"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.config import SCORING_WEIGHTS, MACRO_SCORE_MAX, MACRO_SCORE_MIN
from src.data.data_loader import MarketData
from src.data.margin_client import MarginClient
from src.screening.fundamental import calculate_fundamental_score
from src.screening.news_filter import NewsFilter
from src.screening.pipeline import CandidateStock
from src.notify.formatter import ScoredCandidate
from src.utils.logging_config import logger


class MultiFactorScorer:
    """マルチファクタースコアリング

    各ファクターを0-100で評価し、ウェイト加重平均で最終スコアを算出。
    ニュース・開示スコアとマクロ環境補正を加える。
    """

    def __init__(self) -> None:
        self._news_filter = NewsFilter()
        self._margin_client = MarginClient()
        self._weights = SCORING_WEIGHTS

    def score(
        self,
        candidates: list[CandidateStock],
        market_data: MarketData,
        direction: str = "buy",
    ) -> list[ScoredCandidate]:
        """候補銘柄リストをスコアリング

        Args:
            candidates: スクリーニング通過した候補
            market_data: 全データ
            direction: "buy" or "sell"

        Returns:
            ScoredCandidateのリスト
        """
        scored = []
        macro_score = self._news_filter.get_macro_score(market_data)

        # 信用残データを一括取得（候補銘柄のみ）
        codes = [c.code for c in candidates]
        margin_data = pd.DataFrame()
        try:
            margin_data = self._margin_client.fetch_margin_for_codes(codes[:30])
        except Exception as e:
            logger.debug(f"信用残データ一括取得失敗: {e}")

        for candidate in candidates:
            try:
                sc = self._score_single(
                    candidate, market_data, direction, macro_score, margin_data
                )
                scored.append(sc)
            except Exception as e:
                logger.warning(f"スコアリング失敗 ({candidate.code}): {e}")

        return scored

    def _score_single(
        self,
        candidate: CandidateStock,
        market_data: MarketData,
        direction: str,
        macro_score: float,
        margin_data: pd.DataFrame,
    ) -> ScoredCandidate:
        """単一銘柄のスコアリング"""
        sc = ScoredCandidate(
            code=candidate.code,
            name=candidate.name,
            close=candidate.close,
            direction=direction,
            signals=candidate.signals,
        )

        df = candidate.prices_df
        if df.empty:
            return sc

        last = df.iloc[-1]
        w = self._weights

        # === 1. トレンドスコア ===
        sc.trend_score = self._calc_trend_score(df, last, direction)

        # === 2. MACDスコア ===
        sc.macd_score = self._calc_macd_score(df, last, direction)

        # === 3. 出来高スコア ===
        sc.volume_score = self._calc_volume_score(df, last)

        # === 4. ファンダメンタルスコア ===
        fund_score, fund_details = calculate_fundamental_score(
            candidate.code, market_data.financials, candidate.close
        )
        sc.fundamental_score = fund_score
        sc.per = fund_details.get("per")
        sc.pbr = fund_details.get("pbr")
        sc.roe = fund_details.get("roe")

        # === 5. RSIスコア ===
        sc.rsi_score = self._calc_rsi_score(last, direction)

        # === 6. 一目均衡表スコア ===
        sc.ichimoku_score = self._calc_ichimoku_score(last, direction)

        # === 7. パターンスコア ===
        sc.pattern_score = self._calc_pattern_score(candidate.signals, direction)

        # === 8. リスク/リワードスコア ===
        sc.risk_reward_score = self._calc_risk_reward_score(df, last, direction)

        # === 9. ニュース・開示スコア ===
        disclosure_score, disclosure_summary = self._news_filter.calculate_disclosure_score(
            candidate.code, market_data
        )
        sentiment_score, sentiment_label, news_summary = (
            self._news_filter.calculate_news_sentiment(
                candidate.code, candidate.name, market_data
            )
        )
        sc.news_score = min(100, max(0, 50 + disclosure_score + sentiment_score * 3))
        sc.news_summary = news_summary or disclosure_summary
        sc.news_sentiment = sentiment_label

        # === 10. 需給スコア ===
        sc.margin_score = self._calc_margin_score(candidate.code, margin_data, direction)

        # === 加重平均 ===
        base_score = (
            sc.trend_score * w.trend
            + sc.macd_score * w.macd
            + sc.volume_score * w.volume
            + sc.fundamental_score * w.fundamental
            + sc.rsi_score * w.rsi
            + sc.ichimoku_score * w.ichimoku
            + sc.pattern_score * w.pattern
            + sc.risk_reward_score * w.risk_reward
            + sc.news_score * w.news_disclosure
            + sc.margin_score * w.margin_supply
        )

        # マクロ環境補正
        sc.macro_adjustment = macro_score
        sc.total_score = max(0, min(100, base_score + macro_score))

        return sc

    def _calc_trend_score(self, df: pd.DataFrame, last: pd.Series, direction: str) -> float:
        """トレンドスコア（0-100）"""
        score = 50.0

        # SMA配列
        sma5 = last.get("SMA_5")
        sma25 = last.get("SMA_25")
        sma75 = last.get("SMA_75")
        close = last.get("Close", 0)

        if pd.notna(sma5) and pd.notna(sma25):
            if direction == "buy":
                if close > sma5 > sma25:
                    score += 25
                    if pd.notna(sma75) and sma25 > sma75:
                        score += 15  # パーフェクトオーダー
                elif close > sma5:
                    score += 10
            else:  # sell
                if close < sma5 < sma25:
                    score += 25
                    if pd.notna(sma75) and sma25 < sma75:
                        score += 15
                elif close < sma5:
                    score += 10

        return min(100, max(0, score))

    def _calc_macd_score(self, df: pd.DataFrame, last: pd.Series, direction: str) -> float:
        """MACDスコア（0-100）"""
        score = 50.0

        macd = last.get("MACD")
        signal = last.get("MACDs")
        hist = last.get("MACDh")

        if pd.notna(macd) and pd.notna(signal):
            if direction == "buy":
                if macd > signal:
                    score += 20
                if pd.notna(hist) and hist > 0:
                    score += 10
                    # ヒストグラム増加中
                    if len(df) >= 2:
                        prev_hist = df.iloc[-2].get("MACDh", 0)
                        if pd.notna(prev_hist) and hist > prev_hist:
                            score += 10
            else:
                if macd < signal:
                    score += 20
                if pd.notna(hist) and hist < 0:
                    score += 10

        return min(100, max(0, score))

    def _calc_volume_score(self, df: pd.DataFrame, last: pd.Series) -> float:
        """出来高スコア（0-100）"""
        score = 50.0

        vol_ratio = last.get("VolumeRatio")
        if pd.notna(vol_ratio):
            if vol_ratio > 2.0:
                score += 30
            elif vol_ratio > 1.5:
                score += 20
            elif vol_ratio > 1.2:
                score += 10

        # OBVトレンド
        if "OBV" in df.columns and len(df) >= 10:
            obv_recent = df["OBV"].tail(10)
            if obv_recent.is_monotonic_increasing:
                score += 10
            elif obv_recent.iloc[-1] > obv_recent.iloc[0]:
                score += 5

        return min(100, max(0, score))

    def _calc_rsi_score(self, last: pd.Series, direction: str) -> float:
        """RSIスコア（0-100）"""
        score = 50.0
        rsi = last.get("RSI")

        if pd.notna(rsi):
            if direction == "buy":
                if 30 <= rsi <= 45:
                    score += 30  # 売られすぎから回復
                elif 45 < rsi <= 60:
                    score += 15  # 中立〜やや強
                elif rsi > 70:
                    score -= 15  # 買われすぎ
            else:
                if 55 <= rsi <= 70:
                    score += 30
                elif rsi > 70:
                    score += 15
                elif rsi < 30:
                    score -= 15

        return min(100, max(0, score))

    def _calc_ichimoku_score(self, last: pd.Series, direction: str) -> float:
        """一目均衡表スコア（0-100）"""
        score = 50.0
        close = last.get("Close", 0)

        tenkan = last.get("ISA_9")  # 転換線
        kijun = last.get("ISB_26")   # 基準線
        senkou_a = last.get("ITS_9")  # 先行スパンA
        senkou_b = last.get("IKS_26")  # 先行スパンB

        if direction == "buy":
            if pd.notna(tenkan) and pd.notna(kijun):
                if tenkan > kijun:
                    score += 10
            if pd.notna(senkou_a) and pd.notna(senkou_b):
                cloud_top = max(senkou_a, senkou_b)
                if close > cloud_top:
                    score += 20  # 雲上
                elif close > min(senkou_a, senkou_b):
                    score += 5   # 雲中
        else:
            if pd.notna(tenkan) and pd.notna(kijun):
                if tenkan < kijun:
                    score += 10
            if pd.notna(senkou_a) and pd.notna(senkou_b):
                cloud_bottom = min(senkou_a, senkou_b)
                if close < cloud_bottom:
                    score += 20

        return min(100, max(0, score))

    def _calc_pattern_score(self, signals: list[str], direction: str) -> float:
        """パターンスコア（0-100）"""
        score = 50.0
        buy_patterns = ["包み足(強気)", "はらみ足(強気)", "ハンマー", "三兵"]
        sell_patterns = ["包み足(弱気)", "はらみ足(弱気)", "シューティングスター", "三羽烏"]

        target = buy_patterns if direction == "buy" else sell_patterns
        for pattern in target:
            if any(pattern in s for s in signals):
                score += 15

        return min(100, max(0, score))

    def _calc_risk_reward_score(
        self, df: pd.DataFrame, last: pd.Series, direction: str
    ) -> float:
        """リスク/リワードスコア（0-100）"""
        score = 50.0
        atr = last.get("ATR")
        close = last.get("Close", 0)

        if pd.notna(atr) and atr > 0 and close > 0:
            # ATR対価格比率（ボラティリティが適度か）
            atr_ratio = atr / close
            if 0.01 <= atr_ratio <= 0.04:
                score += 20  # 適度なボラティリティ
            elif atr_ratio > 0.06:
                score -= 10  # ボラティリティ過大

        return min(100, max(0, score))

    def _calc_margin_score(
        self, code: str, margin_data: pd.DataFrame, direction: str
    ) -> float:
        """需給（信用残）スコア（0-100）"""
        score = 50.0

        if margin_data.empty or "code" not in margin_data.columns:
            return score

        row = margin_data[margin_data["code"] == code]
        if row.empty:
            return score

        margin_ratio = row.iloc[0].get("margin_ratio")
        if pd.notna(margin_ratio):
            if direction == "buy":
                if margin_ratio < 1.0:
                    score += 25  # 売り残多い = 踏み上げ期待
                elif margin_ratio < 2.0:
                    score += 15
                elif margin_ratio > 5.0:
                    score -= 15  # 買い残過多
            else:
                if margin_ratio > 5.0:
                    score += 20  # 買い残過多 = 売り圧力
                elif margin_ratio > 3.0:
                    score += 10

        return min(100, max(0, score))
