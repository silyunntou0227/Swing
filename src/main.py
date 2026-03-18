"""日本株スイングトレード自動スクリーニングシステム — エントリーポイント"""

from __future__ import annotations

import argparse
import datetime
import sys
import time

from src.utils.logging_config import logger
from src.data.data_loader import DataLoader
from src.screening.pipeline import ScreeningPipeline
from src.scoring.scorer import MultiFactorScorer
from src.scoring.risk import RiskCalculator
from src.notify.discord import DiscordNotifier
from src.notify.line import LINENotifier
from src.notify.formatter import ResultFormatter, LINEResultFormatter
from src.config import (
    DISCORD_WEBHOOK_URL,
    TOP_BUY_CANDIDATES,
    TOP_SELL_CANDIDATES,
    LINE_CHANNEL_TOKEN,
    LINE_USER_ID,
)


def _send_formatted(notifier: DiscordNotifier, data: dict | str) -> bool:
    """formatter が返す dict/str を適切に Discord に送信。

    Returns:
        全送信が成功した場合 True
    """
    results: list[bool] = []
    if isinstance(data, str):
        results.append(notifier.send(content=data))
    elif isinstance(data, dict) and "embeds" in data:
        for embed in data["embeds"]:
            results.append(notifier.send(embed=embed))
    elif isinstance(data, dict):
        results.append(notifier.send(embed=data))
    return all(results) if results else True


def _is_jpx_holiday(dt: datetime.date) -> bool:
    """東証の休場日（土日 + 日本の祝日）かどうかを判定する。

    jpholiday パッケージが利用可能な場合はそれを使い、
    利用不可ならば土日のみチェックする。
    """
    # 土日チェック
    if dt.weekday() >= 5:
        return True
    # 祝日チェック
    try:
        import jpholiday  # type: ignore[import-untyped]

        if jpholiday.is_holiday(dt):
            return True
    except ImportError:
        logger.debug("jpholiday 未インストール — 祝日チェックをスキップ")
    return False


def main(channel: str = "all") -> int:
    start_time = time.time()
    logger.info("=== 日本株スイングトレードスキャン開始 ===")

    # 東証休場日チェック — 休場日は候補が出ないためスキップ
    today = datetime.date.today()
    if _is_jpx_holiday(today):
        logger.info(f"{today} は東証休場日のためスキャンをスキップします")
        return 0

    try:
        # Step 1: データ取得
        logger.info("Step 1: データ取得中...")
        loader = DataLoader()
        market_data = loader.load_all()
        logger.info(
            f"データ取得完了: {len(market_data.stocks)}銘柄, "
            f"株価データ={market_data.has_prices}, "
            f"財務データ={market_data.has_financials}, "
            f"ニュースデータ={market_data.has_news}"
        )

        # Step 2: 5層スクリーニング
        logger.info("Step 2: スクリーニング実行中...")
        pipeline = ScreeningPipeline()
        candidates = pipeline.run(market_data)
        logger.info(
            f"スクリーニング完了: 買い候補{len(candidates.buy)}件, "
            f"売り候補{len(candidates.sell)}件"
        )

        # Step 3: スコアリング
        logger.info("Step 3: スコアリング実行中...")
        scorer = MultiFactorScorer()
        scored_buy = scorer.score(candidates.buy, market_data, direction="buy")
        scored_sell = scorer.score(candidates.sell, market_data, direction="sell")

        # Top N を選出
        top_buy = sorted(scored_buy, key=lambda x: x.total_score, reverse=True)[
            :TOP_BUY_CANDIDATES
        ]
        top_sell = sorted(scored_sell, key=lambda x: x.total_score, reverse=True)[
            :TOP_SELL_CANDIDATES
        ]
        logger.info(f"スコアリング完了: 買いTop{len(top_buy)}, 売りTop{len(top_sell)}")

        # Step 4: リスク計算
        logger.info("Step 4: リスク計算中...")
        risk_calc = RiskCalculator()
        for candidate in top_buy + top_sell:
            risk_calc.calculate(candidate, market_data)

        # Step 5: Discord 通知送信
        if channel in ("discord", "all"):
            logger.info("Step 5: Discord 通知送信中...")
            formatter = ResultFormatter()
            notifier = DiscordNotifier()

            if not DISCORD_WEBHOOK_URL:
                logger.error("DISCORD_WEBHOOK_URL が未設定です — Discord 通知をスキップ")

            sent_ok = 0
            sent_fail = 0

            # 市場サマリー
            summary = formatter.format_market_summary(market_data)
            if _send_formatted(notifier, summary):
                sent_ok += 1
            else:
                sent_fail += 1

            # 買い候補
            for i, candidate in enumerate(top_buy, 1):
                message = formatter.format_buy_candidate(candidate, rank=i)
                if _send_formatted(notifier, message):
                    sent_ok += 1
                else:
                    sent_fail += 1

            # 売り候補
            for i, candidate in enumerate(top_sell, 1):
                message = formatter.format_sell_candidate(candidate, rank=i)
                if _send_formatted(notifier, message):
                    sent_ok += 1
                else:
                    sent_fail += 1

            # スコアリングサマリー（銘柄一覧 + 推論根拠）
            logger.info("Step 6: スコアリングサマリー送信中...")
            summary_embeds = formatter.format_scoring_summary(top_buy, top_sell)
            for embed in summary_embeds:
                if notifier.send(embed=embed):
                    sent_ok += 1
                else:
                    sent_fail += 1

            if sent_fail > 0:
                logger.warning(
                    f"Discord 通知: 成功{sent_ok}件, 失敗{sent_fail}件"
                )
            else:
                logger.info(f"Discord 通知: 全{sent_ok}件送信完了")
        else:
            logger.info("Step 5: Discord 通知をスキップ (channel=%s)", channel)

        # Step 7: LINE 通知送信（Flex Message + テキストフォールバック）
        if channel in ("line", "all"):
            if LINE_CHANNEL_TOKEN and LINE_USER_ID:
                logger.info("Step 7: LINE 通知送信中...")
                line_formatter = LINEResultFormatter()
                line_notifier = LINENotifier()

                # Flex Message（リッチ表示）を優先送信
                flex_contents = line_formatter.build_flex_summary(
                    market_data, top_buy, top_sell,
                )
                alt_text = line_formatter.format_summary(
                    market_data, top_buy, top_sell,
                )
                if line_notifier.send_flex(alt_text, flex_contents):
                    logger.info("LINE Flex Message 送信完了")
                else:
                    # Flex 失敗時はテキストにフォールバック
                    logger.warning("LINE Flex Message 送信失敗 — テキスト送信にフォールバック")
                    if line_notifier.send(alt_text):
                        logger.info("LINE テキスト通知送信完了")
                    else:
                        logger.warning("LINE 通知の送信に失敗しました")
            else:
                logger.info("Step 7: LINE 認証情報未設定 — LINE 通知をスキップ")
        else:
            logger.info("Step 7: LINE 通知をスキップ (channel=%s)", channel)

        elapsed = time.time() - start_time
        logger.info(f"=== スキャン完了（{elapsed:.1f}秒） ===")
        return 0

    except Exception as e:
        logger.error(f"スキャン中にエラー発生: {e}", exc_info=True)

        # エラー通知
        if channel in ("discord", "all"):
            try:
                notifier = DiscordNotifier()
                notifier.send_error(str(e))
            except Exception:
                logger.error("エラー通知の送信にも失敗しました")

        # LINE にもエラー通知を試行
        if channel in ("line", "all"):
            try:
                if LINE_CHANNEL_TOKEN and LINE_USER_ID:
                    line_notifier = LINENotifier()
                    line_notifier.send(f"【スキャンエラー】\n{str(e)[:4000]}")
            except Exception:
                logger.error("LINE エラー通知の送信にも失敗しました")

        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日本株スイングトレードスキャン")
    parser.add_argument(
        "--channel",
        choices=["discord", "line", "all"],
        default="all",
        help="通知チャンネル (discord/line/all)",
    )
    args = parser.parse_args()
    sys.exit(main(channel=args.channel))
