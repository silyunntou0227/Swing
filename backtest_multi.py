"""5年間×20期間 大規模バックテスト

2021年〜2026年の間からランダムに20の期間を選び、
各期間でスクリーニング→7日後の値動きを検証する。
スコアリング内訳と精度を詳細分析し、改善ポイントを特定する。
"""
from __future__ import annotations

import os
import sys
import time
import random
import json
from datetime import date, timedelta
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from src.data.stock_list import fetch_jpx_stock_list, get_tradeable_codes
from src.screening.pipeline import ScreeningPipeline
from src.scoring.scorer import MultiFactorScorer
from src.data.data_loader import MarketData
from src.data.yahoo_client import YahooClient
from src.config import TOP_BUY_CANDIDATES, TOP_SELL_CANDIDATES, DISCORD_WEBHOOK_URL
from src.utils.logging_config import logger


def send_discord(content: str) -> None:
    """Discord送信"""
    if not DISCORD_WEBHOOK_URL:
        return
    import requests
    try:
        for i in range(0, len(content), 1900):
            requests.post(DISCORD_WEBHOOK_URL, json={"content": content[i:i+1900]}, timeout=10)
            time.sleep(0.5)
    except Exception as e:
        print(f"Discord送信エラー: {e}")


def generate_random_dates(n: int = 20, start_year: int = 2021, end_year: int = 2026) -> list[date]:
    """ランダムな営業日（金曜日 or 平日）を生成"""
    start = date(start_year, 1, 4)
    end = date(end_year, 3, 6)  # 最新データの1週間前
    all_dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # 平日
            all_dates.append(current)
        current += timedelta(days=1)

    # ランダムに選択（最低30日間隔を確保）
    selected = []
    random.seed(42)  # 再現性のため固定シード
    candidates = all_dates.copy()
    random.shuffle(candidates)

    for d in candidates:
        if len(selected) >= n:
            break
        # 既選択の日付と30日以上離れているか
        if all(abs((d - s).days) >= 30 for s in selected):
            selected.append(d)

    return sorted(selected)


def run_single_backtest(
    cutoff_date: date,
    check_days: int,
    stocks: pd.DataFrame,
    codes: list[str],
    yahoo: YahooClient,
) -> dict:
    """単一期間のバックテストを実行"""
    result = {
        "cutoff_date": cutoff_date.isoformat(),
        "check_days": check_days,
        "buy_candidates": [],
        "sell_candidates": [],
        "buy_results": [],
        "sell_results": [],
        "score_details": [],
    }

    # 株価取得（2年分 → cutoff以前に絞る）
    start_date = cutoff_date - timedelta(days=400)
    end_date = cutoff_date + timedelta(days=check_days + 5)

    try:
        prices = yahoo.fetch_bulk_prices(codes, period="2y")
    except Exception as e:
        print(f"    株価取得エラー: {e}")
        return result

    if prices.empty:
        return result

    # 前処理
    if "Date" in prices.columns:
        prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    if "Code" in prices.columns:
        prices["Code"] = prices["Code"].astype(str).str[:4]

    cutoff_dt = pd.Timestamp(cutoff_date)
    prices_before = prices[prices["Date"] <= cutoff_dt].copy()
    prices_before = prices_before.sort_values(["Code", "Date"]).reset_index(drop=True)

    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing = [c for c in price_cols if c in prices_before.columns]
    if existing:
        prices_before = prices_before.dropna(subset=existing)

    if prices_before.empty or prices_before["Code"].nunique() < 100:
        print(f"    データ不足: {prices_before['Code'].nunique() if not prices_before.empty else 0}銘柄")
        return result

    # スクリーニング
    market_data = MarketData(
        stocks=stocks,
        prices=prices_before,
        financials=pd.DataFrame(),
        scan_date=cutoff_date,
    )

    pipeline = ScreeningPipeline()
    candidates = pipeline.run(market_data)

    # スコアリング
    scorer = MultiFactorScorer()
    scored_buy = scorer.score(candidates.buy, market_data, direction="buy")
    scored_sell = scorer.score(candidates.sell, market_data, direction="sell")

    top_buy = sorted(scored_buy, key=lambda x: x.total_score, reverse=True)[:TOP_BUY_CANDIDATES]
    top_sell = sorted(scored_sell, key=lambda x: x.total_score, reverse=True)[:TOP_SELL_CANDIDATES]

    # 検証（推奨保有日数後の値動きをチェック）
    prices_after = prices[prices["Date"] > cutoff_dt].copy()

    for c in top_buy:
        entry_data = prices_before[prices_before["Code"] == c.code]
        if entry_data.empty:
            continue
        entry_price = float(entry_data.iloc[-1]["Close"])

        # 推奨保有日数後の終値
        hold = c.recommended_hold_days if c.recommended_hold_days > 0 else check_days
        after_data = prices_after[prices_after["Code"] == c.code].head(hold)
        if after_data.empty:
            continue

        exit_price = float(after_data.iloc[-1]["Close"])
        pct = (exit_price - entry_price) / entry_price * 100

        # 期間中の最高値・最安値
        max_price = float(after_data["Close"].max()) if not after_data.empty else exit_price
        min_price = float(after_data["Close"].min()) if not after_data.empty else exit_price
        max_pct = (max_price - entry_price) / entry_price * 100
        min_pct = (min_price - entry_price) / entry_price * 100

        result["buy_results"].append(pct)
        result["buy_candidates"].append({
            "code": c.code,
            "name": c.name,
            "score": round(c.total_score, 1),
            "hold_days": hold,
            "entry": round(entry_price, 1),
            "exit": round(exit_price, 1),
            "pct": round(pct, 2),
            "max_pct": round(max_pct, 2),
            "min_pct": round(min_pct, 2),
            "hit": pct > 0,
        })
        result["score_details"].append({
            "code": c.code,
            "direction": "buy",
            "total": round(c.total_score, 1),
            "trend": round(c.trend_score, 1),
            "macd": round(c.macd_score, 1),
            "volume": round(c.volume_score, 1),
            "fundamental": round(c.fundamental_score, 1),
            "rsi": round(c.rsi_score, 1),
            "ichimoku": round(c.ichimoku_score, 1),
            "pattern": round(c.pattern_score, 1),
            "risk_reward": round(c.risk_reward_score, 1),
            "news": round(c.news_score, 1),
            "margin": round(c.margin_score, 1),
            "pct": round(pct, 2),
            "hit": pct > 0,
        })

    for c in top_sell:
        entry_data = prices_before[prices_before["Code"] == c.code]
        if entry_data.empty:
            continue
        entry_price = float(entry_data.iloc[-1]["Close"])

        hold = c.recommended_hold_days if c.recommended_hold_days > 0 else check_days
        after_data = prices_after[prices_after["Code"] == c.code].head(hold)
        if after_data.empty:
            continue

        exit_price = float(after_data.iloc[-1]["Close"])
        pct = (exit_price - entry_price) / entry_price * 100

        result["sell_results"].append(pct)
        result["sell_candidates"].append({
            "code": c.code,
            "name": c.name,
            "score": round(c.total_score, 1),
            "pct": round(pct, 2),
            "hit": pct < 0,
        })
        result["score_details"].append({
            "code": c.code,
            "direction": "sell",
            "total": round(c.total_score, 1),
            "trend": round(c.trend_score, 1),
            "macd": round(c.macd_score, 1),
            "rsi": round(c.rsi_score, 1),
            "ichimoku": round(c.ichimoku_score, 1),
            "pct": round(pct, 2),
            "hit": pct < 0,
        })

    return result


def main():
    t0 = time.time()
    check_days = 7  # デフォルト検証期間

    print("=" * 60)
    print("大規模バックテスト: 5年間×20期間")
    print("=" * 60)

    # 銘柄一覧
    print("\n[1/3] 銘柄一覧取得...")
    stocks = fetch_jpx_stock_list()
    codes = get_tradeable_codes(stocks, max_stocks=800)  # バックテスト用に絞る
    print(f"  対象: {len(codes)}銘柄")

    yahoo = YahooClient()

    # ランダム期間生成
    dates = generate_random_dates(20)
    print(f"\n[2/3] テスト期間: {len(dates)}期間")
    for i, d in enumerate(dates):
        print(f"  {i+1}. {d} ({d.strftime('%a')})")

    # 各期間でバックテスト実行
    print(f"\n[3/3] バックテスト実行中...")
    all_results = []

    for i, cutoff in enumerate(dates):
        elapsed = time.time() - t0
        print(f"\n--- 期間 {i+1}/{len(dates)}: {cutoff} [{elapsed:.0f}s経過] ---")

        result = run_single_backtest(cutoff, check_days, stocks, codes, yahoo)
        all_results.append(result)

        buy_n = len(result["buy_results"])
        buy_wins = sum(1 for r in result["buy_results"] if r > 0)
        sell_n = len(result["sell_results"])
        sell_wins = sum(1 for r in result["sell_results"] if r < 0)

        buy_avg = sum(result["buy_results"]) / len(result["buy_results"]) if result["buy_results"] else 0
        print(f"  買い: {buy_wins}/{buy_n}的中 平均{buy_avg:+.2f}% | 売り: {sell_wins}/{sell_n}的中")

    # ===== 総合分析 =====
    print("\n" + "=" * 60)
    print("総合分析レポート")
    print("=" * 60)

    all_buy_results = []
    all_sell_results = []
    all_score_details = []

    for r in all_results:
        all_buy_results.extend(r["buy_results"])
        all_sell_results.extend(r["sell_results"])
        all_score_details.extend(r["score_details"])

    # 全体パフォーマンス
    if all_buy_results:
        total_buy_wins = sum(1 for r in all_buy_results if r > 0)
        total_buy_avg = sum(all_buy_results) / len(all_buy_results)
        print(f"\n買い候補 全体:")
        print(f"  トレード数: {len(all_buy_results)}")
        print(f"  的中率: {total_buy_wins}/{len(all_buy_results)} ({total_buy_wins/len(all_buy_results)*100:.1f}%)")
        print(f"  平均騰落率: {total_buy_avg:+.2f}%")
        print(f"  最大利益: {max(all_buy_results):+.2f}%")
        print(f"  最大損失: {min(all_buy_results):+.2f}%")

    if all_sell_results:
        total_sell_wins = sum(1 for r in all_sell_results if r < 0)
        total_sell_avg = sum(all_sell_results) / len(all_sell_results)
        print(f"\n売り候補 全体:")
        print(f"  トレード数: {len(all_sell_results)}")
        print(f"  的中率: {total_sell_wins}/{len(all_sell_results)} ({total_sell_wins/len(all_sell_results)*100:.1f}%)")
        print(f"  平均騰落率: {total_sell_avg:+.2f}%")

    # スコア内訳と的中率の相関分析
    if all_score_details:
        buy_details = [d for d in all_score_details if d["direction"] == "buy"]
        if buy_details:
            df_scores = pd.DataFrame(buy_details)

            print(f"\nスコア内訳分析（買い候補 {len(buy_details)}件）:")
            print("-" * 50)

            # 各スコア要因と的中率の関係
            score_factors = ["trend", "macd", "volume", "rsi", "ichimoku", "pattern", "risk_reward"]
            for factor in score_factors:
                if factor not in df_scores.columns:
                    continue
                # スコアが高い群 vs 低い群
                median = df_scores[factor].median()
                high_group = df_scores[df_scores[factor] >= median]
                low_group = df_scores[df_scores[factor] < median]

                high_hit = high_group["hit"].mean() * 100 if len(high_group) > 0 else 0
                low_hit = low_group["hit"].mean() * 100 if len(low_group) > 0 else 0
                high_avg = high_group["pct"].mean() if len(high_group) > 0 else 0
                low_avg = low_group["pct"].mean() if len(low_group) > 0 else 0

                diff = high_hit - low_hit
                indicator = "+++" if diff > 15 else "++" if diff > 5 else "+" if diff > 0 else "-" if diff > -5 else "--"

                print(
                    f"  {factor:12s}: "
                    f"高スコア群 {high_hit:.0f}%({high_avg:+.1f}%) "
                    f"/ 低スコア群 {low_hit:.0f}%({low_avg:+.1f}%) "
                    f"差 {diff:+.0f}% [{indicator}]"
                )

            # スコア帯別の的中率
            print(f"\nスコア帯別の的中率:")
            print("-" * 50)
            bins = [(60, 65), (65, 70), (70, 75), (75, 80), (80, 100)]
            for lo, hi in bins:
                band = df_scores[(df_scores["total"] >= lo) & (df_scores["total"] < hi)]
                if len(band) > 0:
                    hit_rate = band["hit"].mean() * 100
                    avg_ret = band["pct"].mean()
                    print(f"  スコア {lo}-{hi}: {len(band)}件 的中率{hit_rate:.0f}% 平均{avg_ret:+.2f}%")

    # 期間別パフォーマンス
    print(f"\n期間別パフォーマンス:")
    print("-" * 50)
    for r in all_results:
        d = r["cutoff_date"]
        buy_n = len(r["buy_results"])
        buy_wins = sum(1 for x in r["buy_results"] if x > 0)
        buy_avg = sum(r["buy_results"]) / buy_n if buy_n > 0 else 0
        hit_rate = buy_wins / buy_n * 100 if buy_n > 0 else 0
        print(f"  {d}: 買い {buy_wins}/{buy_n} ({hit_rate:.0f}%) 平均{buy_avg:+.2f}%")

    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.0f}秒")

    # Discord通知（サマリーのみ）
    if all_buy_results:
        summary = (
            f"**大規模バックテスト結果 (5年間×{len(dates)}期間)**\n"
            f"買い: {total_buy_wins}/{len(all_buy_results)}的中 "
            f"({total_buy_wins/len(all_buy_results)*100:.0f}%) "
            f"平均{total_buy_avg:+.2f}%\n"
        )
        if all_sell_results:
            summary += (
                f"売り: {total_sell_wins}/{len(all_sell_results)}的中 "
                f"({total_sell_wins/len(all_sell_results)*100:.0f}%) "
                f"平均{total_sell_avg:+.2f}%\n"
            )
        # スコア帯分析も追加
        if all_score_details:
            buy_details = [d for d in all_score_details if d["direction"] == "buy"]
            if buy_details:
                df_s = pd.DataFrame(buy_details)
                summary += "\nスコア帯別的中率:\n"
                for lo, hi in [(60, 65), (65, 70), (70, 75), (75, 80), (80, 100)]:
                    band = df_s[(df_s["total"] >= lo) & (df_s["total"] < hi)]
                    if len(band) > 0:
                        summary += f"  {lo}-{hi}: {len(band)}件 {band['hit'].mean()*100:.0f}%\n"

        send_discord(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
