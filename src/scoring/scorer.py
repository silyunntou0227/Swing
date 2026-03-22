"""加重マルチファクタースコアリングモジュール (0-100)"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.config import (
    SCORING_WEIGHTS, MACRO_SCORE_MAX, MACRO_SCORE_MIN,
    MAX_HOLDING_DAYS, TRAILING_STOP_ATR_MULT, PROFIT_TARGET_ATR_MULT,
    PARTIAL_EXIT_ATR_MULT,
    VOLUME_SCORE_HIGH, VOLUME_SCORE_MID, VOLUME_SCORE_LOW,
    RSI2_EXTREME_OVERSOLD, RSI2_OVERSOLD, RSI2_MILD_OVERSOLD,
    RSI2_MILD_OVERBOUGHT, RSI2_OVERBOUGHT, RSI2_EXTREME_OVERBOUGHT,
    ATR_RATIO_OPTIMAL_MIN, ATR_RATIO_OPTIMAL_MAX, ATR_RATIO_EXCESSIVE,
    HOLD_DAYS_MEAN_REVERSION, HOLD_DAYS_BREAKOUT, HOLD_DAYS_MOMENTUM,
    HOLD_DAYS_DEFAULT, BUY_PATTERNS, SELL_PATTERNS,
)
from src.data.data_loader import MarketData
from src.data.margin_client import MarginClient
from src.screening.fundamental import calculate_fundamental_score
from src.screening.news_filter import NewsFilter
from src.screening.pipeline import CandidateStock
from src.notify.formatter import ScoredCandidate
from src.sector.sector_analyzer import (
    estimate_market_phase,
    calculate_sector_score,
    resolve_sector_for_stock,
)
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

        # 景気サイクル局面を推定
        market_phase = estimate_market_phase(market_data.macro_indicators)
        logger.info(f"景気サイクル判定: {market_phase.value}")

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
                    candidate, market_data, direction, macro_score,
                    margin_data, market_phase,
                )
                scored.append(sc)
            except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
                logger.warning(f"スコアリング失敗 ({candidate.code}): {type(e).__name__}: {e}")

        return scored

    def _score_single(
        self,
        candidate: CandidateStock,
        market_data: MarketData,
        direction: str,
        macro_score: float,
        margin_data: pd.DataFrame,
        market_phase: "MarketPhase | None" = None,
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

        # === 5. RSIスコア（連続下落日数考慮）===
        sc.rsi_score = self._calc_rsi_score(last, direction, df)

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

        # === 11. セクタースコア ===
        sc.sector_score = 50.0  # デフォルト: 中立
        if market_phase is not None:
            sector33 = resolve_sector_for_stock(candidate.code, market_data.stocks)
            if sector33:
                adj, explanation = calculate_sector_score(sector33, market_phase)
                # -10〜+10 のスコア補正を 0-100 スケールに変換
                sc.sector_score = max(0, min(100, 50 + adj * 5))
                sc.sector_explanation = explanation
                from src.sector.sector_config import get_topix17_sector
                sc.sector_name = get_topix17_sector(sector33) or ""

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
            + sc.sector_score * w.sector
        )

        # マクロ環境補正
        sc.macro_adjustment = macro_score
        sc.total_score = max(0, min(100, base_score + macro_score))

        # === 保有期間予測 ===
        self._predict_holding_period(sc, df, last, direction)

        return sc

    def _calc_trend_score(self, df: pd.DataFrame, last: pd.Series, direction: str) -> float:
        """トレンドスコア（0-100）

        日本市場では平均回帰が優位のため、トレンドスコアは
        「買ってはいけない状況」の検出に重点を置く。
        SMA200未満での買いは大きく減点。
        """
        score = 50.0

        sma5 = last.get("SMA_5")
        sma25 = last.get("SMA_25")
        sma75 = last.get("SMA_75")
        sma200 = last.get("SMA_200")
        close = last.get("Close", 0)

        if pd.notna(sma5) and pd.notna(sma25):
            if direction == "buy":
                if close > sma5 > sma25:
                    score += 20
                    if pd.notna(sma75) and sma25 > sma75:
                        score += 10  # パーフェクトオーダー（25+15→20+10に縮小）
                elif close > sma5:
                    score += 8
                # SMA200未満での買いは大幅減点
                if pd.notna(sma200) and close < sma200:
                    score -= 20
            else:
                if close < sma5 < sma25:
                    score += 20
                    if pd.notna(sma75) and sma25 < sma75:
                        score += 10
                elif close < sma5:
                    score += 8
                if pd.notna(sma200) and close > sma200:
                    score -= 15

        return min(100, max(0, score))

    def _calc_macd_score(self, df: pd.DataFrame, last: pd.Series, direction: str) -> float:
        """MACDスコア（0-100）"""
        score = 50.0

        macd = last.get("MACD")
        signal = last.get("MACD_Signal")
        hist = last.get("MACD_Hist")

        if pd.notna(macd) and pd.notna(signal):
            if direction == "buy":
                if macd > signal:
                    score += 20
                if pd.notna(hist) and hist > 0:
                    score += 10
                    # ヒストグラム増加中
                    if len(df) >= 2:
                        prev_hist = df.iloc[-2].get("MACD_Hist", 0)
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
            if vol_ratio > VOLUME_SCORE_HIGH:
                score += 30
            elif vol_ratio > VOLUME_SCORE_MID:
                score += 20
            elif vol_ratio > VOLUME_SCORE_LOW:
                score += 10

        # OBVトレンド
        if "OBV" in df.columns and len(df) >= 10:
            obv_recent = df["OBV"].tail(10)
            if obv_recent.is_monotonic_increasing:
                score += 10
            elif obv_recent.iloc[-1] > obv_recent.iloc[0]:
                score += 5

        return min(100, max(0, score))

    def _calc_rsi_score(self, last: pd.Series, direction: str, df: pd.DataFrame = None) -> float:
        """RSIスコア（0-100）— RSI(14) + Connors RSI(2) + SMA200ゲート

        日本市場の平均回帰特性を活用:
        - RSI(2) < 10 AND Close > SMA200 = 最高確率エントリー（Connors研究: 91%勝率）
        - RSI(2)単独のシグナルはSMA200確認なしでも有効だが減点
        - 連続下落日数で平均回帰の信頼度を強化
        """
        score = 50.0
        rsi14 = last.get("RSI")
        rsi2 = last.get("RSI_short")
        close = last.get("Close", 0)
        sma200 = last.get("SMA_200")

        # SMA200ゲート: 長期上昇トレンド内の押し目買いが最も有効
        above_sma200 = pd.notna(sma200) and close > sma200

        # --- RSI(14): トレンド方向の確認 ---
        if pd.notna(rsi14):
            if direction == "buy":
                if 30 <= rsi14 <= 50:
                    score += 12  # 売られすぎから回復（上限60→50に引下げ）
                elif 50 < rsi14 <= 60:
                    score += 4   # 中立（6→4に減点）
                elif rsi14 > 70:
                    score -= 12  # 買われすぎ（-8→-12に強化）
            else:
                if 55 <= rsi14 <= 70:
                    score += 12
                elif rsi14 > 70:
                    score += 6
                elif rsi14 < 30:
                    score -= 12

        # --- RSI(2): 短期エントリータイミング + SMA200ゲート ---
        if pd.notna(rsi2):
            if direction == "buy":
                if rsi2 < RSI2_EXTREME_OVERSOLD:
                    # SMA200上: 最強シグナル（+35）、SMA200下: 弱シグナル（+10）
                    score += 35 if above_sma200 else 10
                elif rsi2 < RSI2_OVERSOLD:
                    score += 25 if above_sma200 else 8
                elif rsi2 < RSI2_MILD_OVERSOLD:
                    score += 12 if above_sma200 else 4
                elif rsi2 > RSI2_OVERBOUGHT:
                    score -= 15  # 買われすぎペナルティ強化
                elif rsi2 > 30:
                    score -= 10  # 中立ゾーン: 平均回帰シグナルなし → ペナルティ
            else:
                if rsi2 > RSI2_EXTREME_OVERBOUGHT:
                    score += 35 if not above_sma200 else 10
                elif rsi2 > RSI2_OVERBOUGHT:
                    score += 25 if not above_sma200 else 8
                elif rsi2 > RSI2_MILD_OVERBOUGHT:
                    score += 12 if not above_sma200 else 4
                elif rsi2 < RSI2_OVERSOLD:
                    score -= 15
                elif rsi2 < 70:
                    score -= 10  # 中立ゾーン: 平均回帰シグナルなし → ペナルティ

        # --- 連続下落日数ボーナス（平均回帰の信頼度強化）---
        if df is not None and len(df) >= 5 and direction == "buy":
            closes = df["Close"].tail(6).values
            consec_down = 0
            for i in range(len(closes) - 1, 0, -1):
                if closes[i] < closes[i - 1]:
                    consec_down += 1
                else:
                    break
            if consec_down >= 5:
                score += 15
            elif consec_down >= 4:
                score += 12
            elif consec_down >= 3:
                score += 8

        return min(100, max(0, score))

    def _calc_ichimoku_score(self, last: pd.Series, direction: str) -> float:
        """一目均衡表スコア（0-100）"""
        score = 50.0
        close = last.get("Close", 0)

        tenkan = last.get("Ichimoku_Tenkan")  # 転換線
        kijun = last.get("Ichimoku_Kijun")   # 基準線
        senkou_a = last.get("Ichimoku_SenkouA")  # 先行スパンA
        senkou_b = last.get("Ichimoku_SenkouB")  # 先行スパンB

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
        target = BUY_PATTERNS if direction == "buy" else SELL_PATTERNS
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
            if ATR_RATIO_OPTIMAL_MIN <= atr_ratio <= ATR_RATIO_OPTIMAL_MAX:
                score += 20  # 適度なボラティリティ
            elif atr_ratio > ATR_RATIO_EXCESSIVE:
                score -= 10  # ボラティリティ過大

        return min(100, max(0, score))

    def _predict_holding_period(
        self,
        sc: ScoredCandidate,
        df: pd.DataFrame,
        last: pd.Series,
        direction: str,
    ) -> None:
        """保有期間予測（ATRベース + シグナルタイプ別）

        研究に基づくハイブリッド出口戦略:
        - ATRトレーリングストップ（2.5×ATR）
        - 利確目標（3.0×ATR）
        - 部分利確（1.5×ATR で50%決済）
        - 最大保有10日（グリッドサーチ最適化論文に基づく）
        - RSI正常化で出口（RSI(2)が50を超えたら）
        """
        atr = last.get("ATR")
        close = last.get("Close", 0)
        rsi2 = last.get("RSI_short")

        if not pd.notna(atr) or atr <= 0 or close <= 0:
            sc.recommended_hold_days = 7  # デフォルト
            sc.exit_strategy = "ATRデータなし: 7日で見直し"
            return

        # シグナルタイプ別の推奨保有期間
        signals_str = " ".join(sc.signals)
        is_mean_reversion = (
            pd.notna(rsi2) and (rsi2 < 10 or rsi2 > 90)
            or "RSI2" in signals_str
            or "RSI_oversold" in signals_str
        )
        is_breakout = (
            "雲上抜け" in signals_str
            or "ゴールデンクロス" in signals_str
            or "三役好転" in signals_str
        )
        is_momentum = (
            "MACD" in signals_str
            or "パーフェクトオーダー" in signals_str
        )

        if is_mean_reversion:
            hold_days = HOLD_DAYS_MEAN_REVERSION
            strategy_type = "平均回帰(RSI)"
        elif is_breakout:
            hold_days = HOLD_DAYS_BREAKOUT
            strategy_type = "ブレイクアウト"
        elif is_momentum:
            hold_days = HOLD_DAYS_MOMENTUM
            strategy_type = "モメンタム"
        else:
            hold_days = HOLD_DAYS_DEFAULT
            strategy_type = "標準"

        hold_days = min(hold_days, MAX_HOLDING_DAYS)
        sc.recommended_hold_days = hold_days

        # ATRベースの価格目標
        if direction == "buy":
            sc.trailing_stop_price = round(close - atr * TRAILING_STOP_ATR_MULT, 1)
            sc.partial_exit_price = round(close + atr * PARTIAL_EXIT_ATR_MULT, 1)
            tp = round(close + atr * PROFIT_TARGET_ATR_MULT, 1)
            sl_pct = (sc.trailing_stop_price - close) / close * 100
            tp_pct = (tp - close) / close * 100
        else:
            sc.trailing_stop_price = round(close + atr * TRAILING_STOP_ATR_MULT, 1)
            sc.partial_exit_price = round(close - atr * PARTIAL_EXIT_ATR_MULT, 1)
            tp = round(close - atr * PROFIT_TARGET_ATR_MULT, 1)
            sl_pct = (sc.trailing_stop_price - close) / close * 100
            tp_pct = (tp - close) / close * 100

        sc.exit_strategy = (
            f"{strategy_type} | "
            f"SL {sl_pct:+.1f}% TP {tp_pct:+.1f}% | "
            f"最大{hold_days}日"
        )

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
