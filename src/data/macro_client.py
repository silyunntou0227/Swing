"""日銀統計・マクロ経済指標取得クライアント"""

from __future__ import annotations

import pandas as pd
import requests

from src.utils.logging_config import logger

# 日銀時系列統計データ検索サイト API
BOJ_API_BASE = "https://www.stat-search.boj.or.jp/ssi/mtshtml"


class MacroClient:
    """マクロ経済指標取得クライアント

    主要指標:
    - 日経平均のトレンド（Yahoo Finance経由）
    - 無担保コールレート（日銀）
    - 市場環境の総合判定
    """

    def fetch_indicators(self) -> dict:
        """マクロ経済指標を取得

        Returns:
            {
                "market_trend": "bullish" / "bearish" / "neutral",
                "nikkei_change_5d": float,  # 5日変化率
                "nikkei_change_20d": float,  # 20日変化率
                "macro_score": float,  # -5 〜 +5
            }
        """
        indicators = {
            "market_trend": "neutral",
            "nikkei_change_5d": 0.0,
            "nikkei_change_20d": 0.0,
            "macro_score": 0.0,
        }

        # 日経平均のトレンドを取得
        nikkei_data = self._fetch_nikkei_trend()
        indicators.update(nikkei_data)

        # マクロスコア算出
        indicators["macro_score"] = self._calculate_macro_score(indicators)

        return indicators

    def _fetch_nikkei_trend(self) -> dict:
        """日経平均の短期・中期トレンドを取得"""
        try:
            import yfinance as yf

            nikkei = yf.Ticker("^N225")
            hist = nikkei.history(period="3mo")

            if hist.empty or len(hist) < 20:
                return {}

            close = hist["Close"]
            current = close.iloc[-1]

            # 5日変化率
            change_5d = 0.0
            if len(close) >= 6:
                change_5d = (current - close.iloc[-6]) / close.iloc[-6] * 100

            # 20日変化率
            change_20d = 0.0
            if len(close) >= 21:
                change_20d = (current - close.iloc[-21]) / close.iloc[-21] * 100

            # SMA判定
            sma5 = close.tail(5).mean()
            sma25 = close.tail(25).mean() if len(close) >= 25 else sma5

            if current > sma5 > sma25:
                trend = "bullish"
            elif current < sma5 < sma25:
                trend = "bearish"
            else:
                trend = "neutral"

            return {
                "market_trend": trend,
                "nikkei_change_5d": round(change_5d, 2),
                "nikkei_change_20d": round(change_20d, 2),
            }

        except Exception as e:
            logger.warning(f"日経平均トレンド取得失敗: {e}")
            return {}

    def _calculate_macro_score(self, indicators: dict) -> float:
        """マクロ環境スコアを算出（-5〜+5）"""
        score = 0.0

        # 市場トレンド
        trend = indicators.get("market_trend", "neutral")
        if trend == "bullish":
            score += 2.0
        elif trend == "bearish":
            score -= 2.0

        # 短期モメンタム（5日変化率）
        change_5d = indicators.get("nikkei_change_5d", 0.0)
        if change_5d > 2.0:
            score += 1.5
        elif change_5d > 0.5:
            score += 0.5
        elif change_5d < -2.0:
            score -= 1.5
        elif change_5d < -0.5:
            score -= 0.5

        # 中期モメンタム（20日変化率）
        change_20d = indicators.get("nikkei_change_20d", 0.0)
        if change_20d > 5.0:
            score += 1.5
        elif change_20d > 1.0:
            score += 0.5
        elif change_20d < -5.0:
            score -= 1.5
        elif change_20d < -1.0:
            score -= 0.5

        # スコアをクランプ
        return max(-5.0, min(5.0, score))
