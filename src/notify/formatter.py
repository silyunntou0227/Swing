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

    # セクター分析
    sector_name: str = ""  # TOPIX-17セクター名
    sector_score: float = 0.0
    sector_explanation: str = ""

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
    100株単位に切り下げ。100株未満になる場合は資金不足。
    """
    lines = []
    min_lot = 100  # 単元株
    min_cost = min_lot * close

    for capital in POSITION_SIZE_CAPITALS:
        risk_budget = capital * 0.01  # 1%ルール上限
        raw_shares = risk_budget / risk_per_share
        shares = (int(raw_shares) // min_lot) * min_lot  # 100株単位に切り下げ

        label = f"¥{capital // 10000}万"

        # 100株未満 or 購入代金が資金超過 → 資金不足
        if shares < min_lot or min_cost > capital:
            # 100株買った場合の実際のリスクを参考表示
            loss_100 = min_lot * risk_per_share
            loss_pct = loss_100 / capital * 100
            lines.append(
                f"{label}: 1%ルール内で購入不可 "
                f"(100株=¥{min_cost:,.0f}, "
                f"損失¥{loss_100:,.0f}={loss_pct:.1f}%)"
            )
        else:
            cost = shares * close
            max_loss = shares * risk_per_share
            loss_pct = max_loss / capital * 100
            capital_pct = cost / capital * 100
            lines.append(
                f"{label}: **{shares}株** "
                f"(投資額¥{cost:,.0f} = 資金の{capital_pct:.0f}%) "
                f"最大損失¥{max_loss:,.0f}({loss_pct:.1f}%)"
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

        # セクター分析
        if c.sector_name:
            sector_text = f"{c.sector_name}"
            if c.sector_explanation:
                sector_text += f" | {c.sector_explanation}"
            fields.append({
                "name": "セクター分析",
                "value": sector_text,
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

    # ------------------------------------------------------------------
    # スコアリングサマリー（全候補一覧 + 推論根拠）
    # ------------------------------------------------------------------

    def format_scoring_summary(
        self,
        buy_candidates: list[ScoredCandidate],
        sell_candidates: list[ScoredCandidate],
    ) -> list[dict]:
        """全候補の一覧・スコア・推論サマリーを Embed リストで返す。

        Discord Embed は 25 field / 6000文字制限があるため、
        一覧テーブルと個別推論を分割して返す。
        """
        embeds: list[dict] = []

        # --- 買い候補一覧テーブル ---
        if buy_candidates:
            embeds.append(self._build_ranking_table(
                buy_candidates, direction="buy",
            ))
            # 個別推論サマリー（5銘柄ずつ1 Embed）
            for chunk in _chunked(buy_candidates, 5):
                embeds.append(self._build_reasoning_embed(
                    chunk, direction="buy",
                ))

        # --- 売り候補一覧テーブル ---
        if sell_candidates:
            embeds.append(self._build_ranking_table(
                sell_candidates, direction="sell",
            ))
            for chunk in _chunked(sell_candidates, 5):
                embeds.append(self._build_reasoning_embed(
                    chunk, direction="sell",
                ))

        return embeds

    # ---------- 内部ヘルパー ----------

    def _build_ranking_table(
        self,
        candidates: list[ScoredCandidate],
        direction: str,
    ) -> dict:
        """スコア降順のランキングテーブル Embed を生成"""
        emoji = "📈" if direction == "buy" else "📉"
        label = "買い候補" if direction == "buy" else "売り候補"
        color = self.COLOR_BUY if direction == "buy" else self.COLOR_SELL

        lines = [f"```\n{'#':>2} {'銘柄':　<8} {'コード':>6} {'スコア':>5} {'現在値':>9}"]
        lines.append("-" * 44)
        for i, c in enumerate(candidates, 1):
            # 銘柄名は長い場合に切り詰め
            name = c.name[:8] if len(c.name) > 8 else c.name
            lines.append(
                f"{i:>2} {name:　<8} {c.code:>6} "
                f"{c.total_score:>5.0f} ¥{c.close:>9,.0f}"
            )
        lines.append("```")

        return {
            "title": f"{emoji} {label}ランキング（{len(candidates)}件）",
            "description": "\n".join(lines),
            "color": color,
        }

    def _build_reasoning_embed(
        self,
        candidates: list[ScoredCandidate],
        direction: str,
    ) -> dict:
        """各銘柄の推論根拠を field で並べた Embed"""
        color = self.COLOR_BUY if direction == "buy" else self.COLOR_SELL
        fields: list[dict] = []

        for c in candidates:
            reasoning = _build_reasoning_text(c)
            fields.append({
                "name": f"{c.name} ({c.code})  —  {c.total_score:.0f}点",
                "value": reasoning,
                "inline": False,
            })

        label = "買い" if direction == "buy" else "売り"
        return {
            "title": f"🔍 {label}候補 推論サマリー",
            "color": color,
            "fields": fields,
        }


def _build_reasoning_text(c: ScoredCandidate) -> str:
    """1銘柄の推論根拠を簡潔なテキストにまとめる"""
    parts: list[str] = []

    # スコア内訳（上位寄与因子を強調）
    factors = [
        ("トレンド", c.trend_score, 0.20),
        ("MACD", c.macd_score, 0.14),
        ("一目均衡", c.ichimoku_score, 0.12),
        ("出来高", c.volume_score, 0.11),
        ("RSI", c.rsi_score, 0.10),
        ("ファンダ", c.fundamental_score, 0.10),
        ("パターン", c.pattern_score, 0.05),
        ("R:R", c.risk_reward_score, 0.05),
        ("ニュース", c.news_score, 0.05),
        ("セクター", c.sector_score, 0.05),
        ("需給", c.margin_score, 0.03),
    ]
    # 加重スコア降順で上位3因子
    weighted = sorted(factors, key=lambda f: f[1] * f[2], reverse=True)
    top3 = [f"**{name}** {score:.0f}" for name, score, _ in weighted[:3]]
    parts.append(f"主要因子: {' / '.join(top3)}")

    # シグナル
    if c.signals:
        parts.append(f"シグナル: {', '.join(c.signals[:4])}")

    # ファンダメンタル要約
    fund = []
    if c.per is not None:
        fund.append(f"PER {c.per:.1f}")
    if c.pbr is not None:
        fund.append(f"PBR {c.pbr:.2f}")
    if c.roe is not None:
        fund.append(f"ROE {c.roe:.1f}%")
    if fund:
        parts.append(" | ".join(fund))

    # ニュース・セクター
    if c.news_sentiment:
        parts.append(f"ニュース: {c.news_sentiment}")
    if c.sector_name:
        sector_info = c.sector_name
        if c.sector_explanation:
            sector_info += f"（{c.sector_explanation}）"
        parts.append(f"セクター: {sector_info}")

    # マクロ調整
    if c.macro_adjustment != 0:
        parts.append(f"マクロ調整: {c.macro_adjustment:+.0f}点")

    return "\n".join(parts)


def _chunked(lst: list, size: int) -> list[list]:
    """リストを size 個ずつのチャンクに分割"""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


# ============================================================
# LINE 用フォーマッター（プレーンテキスト + Flex Message）
# ============================================================


class LINEResultFormatter:
    """スキャン結果を LINE 向けにフォーマット

    - format_summary: プレーンテキスト版（従来互換）
    - build_flex_summary: Flex Message 版（リッチ表示）
    - build_flex_candidate: 個別銘柄の Flex Message bubble
    """

    # Flex Message カラー
    COLOR_BUY = "#00CC66"
    COLOR_SELL = "#FF4444"
    COLOR_HEADER = "#1A1A2E"
    COLOR_SCORE_HIGH = "#00CC66"
    COLOR_SCORE_MID = "#FFAA00"
    COLOR_SCORE_LOW = "#FF4444"

    def format_summary(
        self,
        market_data: MarketData,
        buy_candidates: list[ScoredCandidate],
        sell_candidates: list[ScoredCandidate],
    ) -> str:
        """LINE 用のサマリーテキストを生成（5000文字以内）"""
        lines: list[str] = []

        # ヘッダ
        lines.append("【日本株スイングスキャン結果】")
        lines.append(f"日付: {market_data.scan_date}")
        lines.append("")

        # 買い候補
        if buy_candidates:
            lines.append(f"▼ 買い候補 TOP{len(buy_candidates)}")
            for i, c in enumerate(buy_candidates, 1):
                sl_pct = (c.stop_loss - c.close) / c.close * 100 if c.stop_loss else 0
                tp_pct = (c.take_profit - c.close) / c.close * 100 if c.take_profit else 0
                lines.append(
                    f"{i}. {c.name}({c.code}) "
                    f"スコア{c.total_score:.0f} ¥{c.close:,.0f}"
                )
                if c.signals:
                    lines.append(f"   シグナル: {', '.join(c.signals[:3])}")
                if c.stop_loss > 0:
                    lines.append(
                        f"   SL ¥{c.stop_loss:,.0f}({sl_pct:+.1f}%) "
                        f"TP ¥{c.take_profit:,.0f}({tp_pct:+.1f}%) "
                        f"R:R 1:{c.risk_reward_ratio:.1f}"
                    )
            lines.append("")

        # 売り候補
        if sell_candidates:
            lines.append(f"▼ 売り候補 TOP{len(sell_candidates)}")
            for i, c in enumerate(sell_candidates, 1):
                lines.append(
                    f"{i}. {c.name}({c.code}) "
                    f"スコア{c.total_score:.0f} ¥{c.close:,.0f}"
                )
                if c.signals:
                    lines.append(f"   シグナル: {', '.join(c.signals[:3])}")
            lines.append("")

        if not buy_candidates and not sell_candidates:
            lines.append("本日は条件を満たす候補銘柄がありませんでした。")

        return "\n".join(lines)[:5000]

    # ------------------------------------------------------------------
    # Flex Message 生成
    # ------------------------------------------------------------------

    def build_flex_summary(
        self,
        market_data: MarketData,
        buy_candidates: list[ScoredCandidate],
        sell_candidates: list[ScoredCandidate],
    ) -> dict:
        """スキャン結果全体を Flex Message carousel として生成

        Returns:
            Flex Message の contents（carousel 型）
        """
        bubbles: list[dict] = []

        # ヘッダー bubble
        bubbles.append(self._build_header_bubble(market_data, buy_candidates, sell_candidates))

        # 買い候補 bubbles（上位5件）
        for i, c in enumerate(buy_candidates[:5], 1):
            bubbles.append(self._build_candidate_bubble(c, rank=i, direction="buy"))

        # 売り候補 bubbles（上位3件）
        for i, c in enumerate(sell_candidates[:3], 1):
            bubbles.append(self._build_candidate_bubble(c, rank=i, direction="sell"))

        # carousel は最大12 bubbles
        return {
            "type": "carousel",
            "contents": bubbles[:12],
        }

    def build_flex_candidate(
        self,
        candidate: ScoredCandidate,
        rank: int,
        direction: str = "buy",
    ) -> dict:
        """個別銘柄の Flex Message bubble を生成

        Returns:
            Flex Message の contents（bubble 型）
        """
        return self._build_candidate_bubble(candidate, rank, direction)

    # ------------------------------------------------------------------
    # Flex bubble builders
    # ------------------------------------------------------------------

    def _build_header_bubble(
        self,
        market_data: MarketData,
        buy_candidates: list[ScoredCandidate],
        sell_candidates: list[ScoredCandidate],
    ) -> dict:
        """サマリーヘッダーの bubble"""
        body_contents: list[dict] = [
            {
                "type": "text",
                "text": "日本株スイングスキャン",
                "weight": "bold",
                "size": "lg",
                "color": self.COLOR_HEADER,
            },
            {
                "type": "text",
                "text": f"スキャン日: {market_data.scan_date}",
                "size": "sm",
                "color": "#888888",
                "margin": "md",
            },
            {"type": "separator", "margin": "lg"},
            {
                "type": "box",
                "layout": "horizontal",
                "margin": "lg",
                "contents": [
                    {
                        "type": "box",
                        "layout": "vertical",
                        "flex": 1,
                        "contents": [
                            {
                                "type": "text",
                                "text": "買い候補",
                                "size": "sm",
                                "color": "#888888",
                                "align": "center",
                            },
                            {
                                "type": "text",
                                "text": str(len(buy_candidates)),
                                "size": "xxl",
                                "weight": "bold",
                                "color": self.COLOR_BUY,
                                "align": "center",
                            },
                        ],
                    },
                    {"type": "separator"},
                    {
                        "type": "box",
                        "layout": "vertical",
                        "flex": 1,
                        "contents": [
                            {
                                "type": "text",
                                "text": "売り候補",
                                "size": "sm",
                                "color": "#888888",
                                "align": "center",
                            },
                            {
                                "type": "text",
                                "text": str(len(sell_candidates)),
                                "size": "xxl",
                                "weight": "bold",
                                "color": self.COLOR_SELL,
                                "align": "center",
                            },
                        ],
                    },
                ],
            },
        ]

        return {
            "type": "bubble",
            "size": "kilo",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": body_contents,
                "paddingAll": "lg",
            },
        }

    def _build_candidate_bubble(
        self,
        c: ScoredCandidate,
        rank: int,
        direction: str,
    ) -> dict:
        """個別銘柄カードの bubble"""
        is_buy = direction == "buy"
        accent = self.COLOR_BUY if is_buy else self.COLOR_SELL
        label = "買い" if is_buy else "売り"

        # スコアに応じた色
        if c.total_score >= 70:
            score_color = self.COLOR_SCORE_HIGH
        elif c.total_score >= 50:
            score_color = self.COLOR_SCORE_MID
        else:
            score_color = self.COLOR_SCORE_LOW

        body_contents: list[dict] = [
            # ヘッダー行: ランク + 銘柄名
            {
                "type": "box",
                "layout": "horizontal",
                "contents": [
                    {
                        "type": "text",
                        "text": f"#{rank} {label}",
                        "size": "xs",
                        "color": "#FFFFFF",
                        "weight": "bold",
                    },
                ],
                "backgroundColor": accent,
                "paddingAll": "xs",
                "cornerRadius": "sm",
            },
            {
                "type": "text",
                "text": f"{c.name}",
                "weight": "bold",
                "size": "md",
                "margin": "md",
            },
            {
                "type": "text",
                "text": c.code,
                "size": "xs",
                "color": "#888888",
            },
            # スコア + 現在値
            {
                "type": "box",
                "layout": "horizontal",
                "margin": "lg",
                "contents": [
                    {
                        "type": "text",
                        "text": f"スコア {c.total_score:.0f}",
                        "weight": "bold",
                        "size": "lg",
                        "color": score_color,
                        "flex": 1,
                    },
                    {
                        "type": "text",
                        "text": f"¥{c.close:,.0f}",
                        "size": "lg",
                        "align": "end",
                        "flex": 1,
                    },
                ],
            },
            {"type": "separator", "margin": "lg"},
        ]

        # シグナル
        if c.signals:
            signals_text = ", ".join(c.signals[:3])
            body_contents.append({
                "type": "text",
                "text": signals_text,
                "size": "xs",
                "color": "#666666",
                "margin": "md",
                "wrap": True,
            })

        # リスク管理（買い候補のみ）
        if c.stop_loss > 0:
            sl_pct = (c.stop_loss - c.close) / c.close * 100
            tp_pct = (c.take_profit - c.close) / c.close * 100
            body_contents.append({
                "type": "box",
                "layout": "vertical",
                "margin": "md",
                "contents": [
                    self._kv_row("損切り", f"¥{c.stop_loss:,.0f} ({sl_pct:+.1f}%)"),
                    self._kv_row("利確", f"¥{c.take_profit:,.0f} ({tp_pct:+.1f}%)"),
                    self._kv_row("R:R", f"1:{c.risk_reward_ratio:.1f}"),
                ],
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
            body_contents.append({
                "type": "text",
                "text": " | ".join(fund_parts),
                "size": "xxs",
                "color": "#999999",
                "margin": "md",
            })

        return {
            "type": "bubble",
            "size": "kilo",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": body_contents,
                "paddingAll": "lg",
            },
        }

    @staticmethod
    def _kv_row(key: str, value: str) -> dict:
        """key-value 行の Flex box"""
        return {
            "type": "box",
            "layout": "horizontal",
            "contents": [
                {
                    "type": "text",
                    "text": key,
                    "size": "xxs",
                    "color": "#888888",
                    "flex": 2,
                },
                {
                    "type": "text",
                    "text": value,
                    "size": "xxs",
                    "align": "end",
                    "flex": 3,
                },
            ],
        }
