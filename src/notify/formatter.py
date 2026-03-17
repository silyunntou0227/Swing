"""通知結果フォーマットモジュール"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.data_loader import MarketData


@dataclass
class ScoredCandidate:
    """スコアリング済み候補銘柄"""

    code: str = ""
    name: str = ""
    close: float = 0.0
    total_score: float = 0.0
    direction: str = "buy"  # "buy" or "sell"

    # スコア内訳
    trend_score: float = 0.0
    macd_score: float = 0.0
    volume_score: float = 0.0
    fundamental_score: float = 0.0
    rsi_score: float = 0.0
    ichimoku_score: float = 0.0
    pattern_score: float = 0.0
    risk_reward_score: float = 0.0
    news_score: float = 0.0
    margin_score: float = 0.0
    macro_adjustment: float = 0.0

    # シグナル詳細
    signals: list[str] = field(default_factory=list)

    # ファンダメンタル
    per: float | None = None
    pbr: float | None = None
    roe: float | None = None

    # 需給
    margin_ratio: float | None = None
    short_selling_ratio: float | None = None

    # ニュース
    news_summary: str = ""
    news_sentiment: str = ""

    # 保有期間予測
    recommended_hold_days: int = 0  # 推奨保有日数
    exit_strategy: str = ""  # 出口戦略の説明
    partial_exit_price: float = 0.0  # 部分利確価格（50%決済）
    trailing_stop_price: float = 0.0  # トレーリングストップ価格

    # リスク管理
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size_shares: int = 0
    position_size_yen: float = 0.0
    risk_reward_ratio: float = 0.0
    ruin_probability: float = 0.0


POSITION_SIZE_CAPITALS = [250_000, 500_000, 1_000_000]  # 25万/50万/100万


def _format_position_sizes(
    close: float, risk_per_share: float, direction: str
) -> str:
    """1%ルールに基づく3パターンのポジションサイズを計算

    1%ルール: 1トレードの最大損失 = 総資金の1%
    株数 = (資金 × 1%) / 1株あたりリスク（損切り幅）
    100株単位に切り下げ
    """
    lines = []
    for capital in POSITION_SIZE_CAPITALS:
        risk_budget = capital * 0.01  # 1%
        raw_shares = risk_budget / risk_per_share
        shares = max(100, (int(raw_shares) // 100) * 100)
        cost = shares * close
        max_loss = shares * risk_per_share
        capital_pct = cost / capital * 100

        label = f"¥{capital:,.0f}"
        if cost > capital:
            lines.append(f"{label}: 資金不足（最低{100}株=¥{100*close:,.0f}）")
        else:
            lines.append(
                f"{label}: **{shares}株** (¥{cost:,.0f}, "
                f"資金の{capital_pct:.0f}%) "
                f"最大損失¥{max_loss:,.0f}"
            )

    return "\n".join(lines)


class ResultFormatter:
    """スキャン結果をDiscord Embed形式にフォーマット"""

    COLOR_BUY = 0x00CC66   # 緑
    COLOR_SELL = 0xFF4444   # 赤
    COLOR_INFO = 0x3399FF   # 青

    def format_market_summary(self, market_data: MarketData) -> dict:
        """市場サマリーをEmbed形式で返す"""
        desc_parts = [f"スキャン日: {market_data.scan_date}"]

        if market_data.has_prices:
            n_stocks = market_data.prices["Code"].nunique()
            desc_parts.append(f"対象銘柄数: {n_stocks}")

        if market_data.has_disclosures:
            desc_parts.append(f"適時開示: {len(market_data.disclosures)}件")

        if market_data.has_news:
            desc_parts.append(f"ニュース: {len(market_data.news)}件")

        if market_data.macro_indicators:
            macro = market_data.macro_indicators
            if "market_trend" in macro:
                desc_parts.append(f"市場トレンド: {macro['market_trend']}")

        return {
            "embeds": [{
                "title": "日本株スイングトレードスキャン",
                "description": "\n".join(desc_parts),
                "color": self.COLOR_INFO,
            }]
        }

    def format_buy_candidate(self, c: ScoredCandidate, rank: int) -> dict:
        """買い候補をEmbed形式で返す"""
        return self._format_candidate(c, rank, direction="buy")

    def format_sell_candidate(self, c: ScoredCandidate, rank: int) -> dict:
        """売り候補をEmbed形式で返す"""
        return self._format_candidate(c, rank, direction="sell")

    def _format_candidate(
        self, c: ScoredCandidate, rank: int, direction: str
    ) -> dict:
        emoji = "📈" if direction == "buy" else "📉"
        label = "買い候補" if direction == "buy" else "売り候補"
        color = self.COLOR_BUY if direction == "buy" else self.COLOR_SELL

        title = f"{emoji} {label} #{rank}: {c.name} ({c.code})"
        desc = f"**スコア: {c.total_score:.0f}/100** | 現在値: ¥{c.close:,.0f}"

        fields = []

        # シグナル
        if c.signals:
            fields.append({
                "name": "シグナル",
                "value": ", ".join(c.signals[:5]),
                "inline": False,
            })

        # トレンド・テクニカル
        tech_parts = []
        if c.trend_score > 0:
            tech_parts.append(f"トレンド: {c.trend_score:.0f}")
        if c.macd_score > 0:
            tech_parts.append(f"MACD: {c.macd_score:.0f}")
        if c.rsi_score > 0:
            tech_parts.append(f"RSI: {c.rsi_score:.0f}")
        if c.ichimoku_score > 0:
            tech_parts.append(f"一目: {c.ichimoku_score:.0f}")
        if tech_parts:
            fields.append({
                "name": "テクニカル",
                "value": " | ".join(tech_parts),
                "inline": True,
            })

        # ファンダメンタル
        fund_parts = []
        if c.per is not None:
            fund_parts.append(f"PER {c.per:.1f}")
        if c.pbr is not None:
            fund_parts.append(f"PBR {c.pbr:.2f}")
        if c.roe is not None:
            fund_parts.append(f"ROE {c.roe:.1f}%")
        if fund_parts:
            fields.append({
                "name": "ファンダメンタル",
                "value": " | ".join(fund_parts),
                "inline": True,
            })

        # 需給
        supply_parts = []
        if c.margin_ratio is not None:
            supply_parts.append(f"信用倍率 {c.margin_ratio:.1f}")
        if c.short_selling_ratio is not None:
            supply_parts.append(f"空売り比率 {c.short_selling_ratio:.0f}%")
        if supply_parts:
            fields.append({
                "name": "需給",
                "value": " | ".join(supply_parts),
                "inline": True,
            })

        # ニュース
        if c.news_summary:
            news_text = f"{c.news_sentiment} | {c.news_summary[:100]}"
            fields.append({
                "name": "📰 ニュース",
                "value": news_text,
                "inline": False,
            })

        # 保有期間予測
        if c.recommended_hold_days > 0:
            hold_parts = [f"推奨保有: {c.recommended_hold_days}日"]
            if c.partial_exit_price > 0:
                hold_parts.append(f"部分利確: ¥{c.partial_exit_price:,.0f}")
            if c.trailing_stop_price > 0:
                hold_parts.append(f"トレーリングSL: ¥{c.trailing_stop_price:,.0f}")
            if c.exit_strategy:
                hold_parts.append(c.exit_strategy)
            fields.append({
                "name": "出口戦略",
                "value": " | ".join(hold_parts),
                "inline": False,
            })

        # リスク管理 + 1%ルール ポジションサイズ提案
        if c.stop_loss > 0:
            sl_pct = (c.stop_loss - c.close) / c.close * 100
            tp_pct = (c.take_profit - c.close) / c.close * 100
            risk_text = (
                f"損切り: ¥{c.stop_loss:,.0f} ({sl_pct:+.1f}%) | "
                f"利確: ¥{c.take_profit:,.0f} ({tp_pct:+.1f}%)\n"
                f"R:R = 1:{c.risk_reward_ratio:.1f} | "
                f"破産確率: {c.ruin_probability:.1f}%"
            )
            fields.append({
                "name": "リスク管理",
                "value": risk_text,
                "inline": False,
            })

            # 1%ルール ポジションサイズ提案（25万/50万/100万の3パターン）
            risk_per_share = abs(c.close - c.stop_loss)
            if risk_per_share > 0:
                pos_text = _format_position_sizes(c.close, risk_per_share, direction)
                fields.append({
                    "name": "1%ルール ポジションサイズ",
                    "value": pos_text,
                    "inline": False,
                })

        return {
            "embeds": [{
                "title": title,
                "description": desc,
                "color": color,
                "fields": fields,
            }]
        }
