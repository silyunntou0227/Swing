"""日本株スイングトレード自動スクリーニングシステム — エントリーポイント"""

from __future__ import annotations

import sys
import time

from src.utils.logging_config import logger
from src.data.data_loader import DataLoader
from src.screening.pipeline import ScreeningPipeline
from src.scoring.scorer import MultiFactorScorer
from src.scoring.risk import RiskCalculator
from src.notify.discord import DiscordNotifier
from src.notify.formatter import ResultFormatter
from src.config import TOP_BUY_CANDIDATES, TOP_SELL_CANDIDATES


def _send_formatted(notifier: DiscordNotifier, data: dict | str) -> None:
    """formatter が返す dict/str を適切に Discord に送信"""
    if isinstance(data, str):
        notifier.send(content=data)
    elif isinstance(data, dict) and "embeds" in data:
        for embed in data["embeds"]:
            notifier.send(embed=embed)
    elif isinstance(data, dict):
        notifier.send(embed=data)


def main() -> int:
    start_time = time.time()
    logger.info("=== 日本株スイングトレードスキャン開始 ===")

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

        # Step 5: 通知送信
        logger.info("Step 5: 通知送信中...")
        formatter = ResultFormatter()
        notifier = DiscordNotifier()

        # 市場サマリー
        summary = formatter.format_market_summary(market_data)
        _send_formatted(notifier, summary)

        # 買い候補
        for i, candidate in enumerate(top_buy, 1):
            message = formatter.format_buy_candidate(candidate, rank=i)
            _send_formatted(notifier, message)

        # 売り候補
        for i, candidate in enumerate(top_sell, 1):
            message = formatter.format_sell_candidate(candidate, rank=i)
            _send_formatted(notifier, message)

        elapsed = time.time() - start_time
        logger.info(f"=== スキャン完了（{elapsed:.1f}秒） ===")
        return 0

    except Exception as e:
        logger.error(f"スキャン中にエラー発生: {e}", exc_info=True)

        # エラー通知
        try:
            notifier = DiscordNotifier()
            notifier.send_error(str(e))
        except Exception:
            logger.error("エラー通知の送信にも失敗しました")

        return 1


if __name__ == "__main__":
    sys.exit(main())
