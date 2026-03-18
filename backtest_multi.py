"""5年間×20期間 大規模バックテスト（最適化版）

株価データを1回だけダウンロードし、各期間はスライスで処理。
GitHub Actions 60分以内に完了する設計。
"""
from __future__ import annotations

import math
import os
import sys
import time
import random
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from src.data.stock_list import fetch_jpx_stock_list, get_tradeable_codes
from src.screening.pipeline import ScreeningPipeline
from src.scoring.scorer import MultiFactorScorer
from src.data.data_loader import MarketData
from src.config import TOP_BUY_CANDIDATES, TOP_SELL_CANDIDATES
from src.utils.logging_config import logger
from src.utils.discord_helper import send_discord_text as send_discord


def generate_random_dates(n: int = 20) -> list[date]:
    """5年間からランダムに営業日を選択（固定シード）"""
    end = date.today() - timedelta(days=10)  # 検証に余裕を持たせる
    start = date(end.year - 5, end.month, end.day)
    all_dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            all_dates.append(current)
        current += timedelta(days=1)

    random.seed(42)
    candidates = all_dates.copy()
    random.shuffle(candidates)

    selected = []
    for d in candidates:
        if len(selected) >= n:
            break
        if all(abs((d - s).days) >= 30 for s in selected):
            selected.append(d)
    return sorted(selected)


def download_all_prices(codes: list[str]) -> pd.DataFrame:
    """全銘柄の5年間株価を1回でダウンロード（最大の最適化ポイント）"""
    tickers = [f"{c}.T" for c in codes]
    chunk_size = 200
    all_frames = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"  ダウンロード: チャンク {i//chunk_size+1}/{(len(tickers)-1)//chunk_size+1} ({len(chunk)}銘柄)")
        try:
            data = yf.download(
                chunk,
                period="5y",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if data.empty:
                continue

            # マルチカラム → ロング形式変換
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in chunk:
                    code = ticker.replace(".T", "")
                    try:
                        df_single = data.xs(ticker, level=1, axis=1).copy()
                        df_single = df_single.reset_index()
                        df_single["Code"] = code
                        df_single = df_single.rename(columns={"index": "Date"})
                        if "Date" not in df_single.columns:
                            df_single = df_single.reset_index()
                            if "Date" not in df_single.columns:
                                for col in df_single.columns:
                                    if "date" in str(col).lower():
                                        df_single = df_single.rename(columns={col: "Date"})
                                        break
                        if not df_single.empty and "Close" in df_single.columns:
                            all_frames.append(df_single[["Date", "Open", "High", "Low", "Close", "Volume", "Code"]].dropna(subset=["Close"]))
                    except (KeyError, TypeError):
                        continue
            else:
                data = data.reset_index()
                if len(chunk) == 1:
                    data["Code"] = chunk[0].replace(".T", "")
                    if "Close" in data.columns:
                        all_frames.append(data[["Date", "Open", "High", "Low", "Close", "Volume", "Code"]].dropna(subset=["Close"]))
        except Exception as e:
            print(f"    チャンクエラー: {e}")
            continue

        time.sleep(1)

    if not all_frames:
        return pd.DataFrame()

    prices = pd.concat(all_frames, ignore_index=True)
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    prices["Code"] = prices["Code"].astype(str).str[:4]
    prices = prices.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"  完了: {prices['Code'].nunique()}銘柄, {len(prices)}行")
    return prices


def run_single_backtest(
    cutoff_date: date,
    all_prices: pd.DataFrame,
    stocks: pd.DataFrame,
    check_days: int = 7,
) -> dict:
    """単一期間のバックテスト（ダウンロード済みデータを使用）"""
    result = {
        "cutoff_date": cutoff_date.isoformat(),
        "buy_results": [],
        "sell_results": [],
        "score_details": [],
    }

    cutoff_dt = pd.Timestamp(cutoff_date)
    # cutoff前300日〜cutoffの株価データ
    start_dt = cutoff_dt - pd.Timedelta(days=400)
    prices_before = all_prices[
        (all_prices["Date"] >= start_dt) & (all_prices["Date"] <= cutoff_dt)
    ].copy()

    if prices_before.empty or prices_before["Code"].nunique() < 50:
        return result

    # cutoff後のデータ（検証用）
    end_dt = cutoff_dt + pd.Timedelta(days=check_days * 2)
    prices_after = all_prices[
        (all_prices["Date"] > cutoff_dt) & (all_prices["Date"] <= end_dt)
    ].copy()

    # スクリーニング
    market_data = MarketData(
        stocks=stocks,
        prices=prices_before,
        financials=pd.DataFrame(),
        scan_date=cutoff_date,
    )

    try:
        pipeline = ScreeningPipeline()
        candidates = pipeline.run(market_data)
    except Exception as e:
        print(f"    スクリーニングエラー: {e}")
        return result

    # スコアリング
    try:
        scorer = MultiFactorScorer()
        scored_buy = scorer.score(candidates.buy, market_data, direction="buy")
        scored_sell = scorer.score(candidates.sell, market_data, direction="sell")
    except Exception as e:
        print(f"    スコアリングエラー: {e}")
        return result

    top_buy = sorted(scored_buy, key=lambda x: x.total_score, reverse=True)[:TOP_BUY_CANDIDATES]
    top_sell = sorted(scored_sell, key=lambda x: x.total_score, reverse=True)[:TOP_SELL_CANDIDATES]

    # 検証
    for c in top_buy:
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
        max_pct = (float(after_data["Close"].max()) - entry_price) / entry_price * 100
        min_pct = (float(after_data["Close"].min()) - entry_price) / entry_price * 100

        result["buy_results"].append(pct)
        result["score_details"].append({
            "code": c.code, "name": c.name, "direction": "buy",
            "total": round(c.total_score, 1),
            "trend": round(c.trend_score, 1),
            "macd": round(c.macd_score, 1),
            "volume": round(c.volume_score, 1),
            "fundamental": round(c.fundamental_score, 1),
            "rsi": round(c.rsi_score, 1),
            "ichimoku": round(c.ichimoku_score, 1),
            "pattern": round(c.pattern_score, 1),
            "risk_reward": round(c.risk_reward_score, 1),
            "sector": round(c.sector_score, 1),
            "sector_name": c.sector_name,
            "pct": round(pct, 2), "max_pct": round(max_pct, 2),
            "min_pct": round(min_pct, 2), "hit": pct > 0,
            "hold_days": hold,
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
        result["score_details"].append({
            "code": c.code, "name": c.name, "direction": "sell",
            "total": round(c.total_score, 1),
            "trend": round(c.trend_score, 1),
            "macd": round(c.macd_score, 1),
            "volume": round(c.volume_score, 1),
            "fundamental": round(c.fundamental_score, 1),
            "rsi": round(c.rsi_score, 1),
            "ichimoku": round(c.ichimoku_score, 1),
            "pattern": round(c.pattern_score, 1),
            "risk_reward": round(c.risk_reward_score, 1),
            "sector": round(c.sector_score, 1),
            "sector_name": c.sector_name,
            "pct": round(pct, 2), "hit": pct < 0,
        })

    return result


def main():
    t0 = time.time()

    print("=" * 60)
    print("大規模バックテスト: 5年間×20期間（最適化版）")
    print("=" * 60)

    # 銘柄一覧
    print("\n[1/4] 銘柄一覧取得...")
    stocks = fetch_jpx_stock_list()
    codes = get_tradeable_codes(stocks, max_stocks=600)
    print(f"  対象: {len(codes)}銘柄")

    # 全株価を1回でダウンロード（最大の時間節約）
    print(f"\n[2/4] 5年間の株価データを一括ダウンロード...")
    all_prices = download_all_prices(codes)
    if all_prices.empty:
        print("ERROR: 株価データ取得失敗")
        return 1
    print(f"  ダウンロード完了: {time.time()-t0:.0f}秒")

    # ランダム期間生成
    dates = generate_random_dates(20)
    print(f"\n[3/4] テスト期間: {len(dates)}期間")
    for i, d in enumerate(dates):
        print(f"  {i+1}. {d} ({d.strftime('%a')})")

    # 各期間でバックテスト
    print(f"\n[4/4] バックテスト実行中...")
    all_results = []

    for i, cutoff in enumerate(dates):
        elapsed = time.time() - t0
        print(f"\n--- 期間 {i+1}/{len(dates)}: {cutoff} [{elapsed:.0f}s] ---")

        result = run_single_backtest(cutoff, all_prices, stocks)
        all_results.append(result)

        buy_n = len(result["buy_results"])
        buy_wins = sum(1 for r in result["buy_results"] if r > 0)
        buy_avg = sum(result["buy_results"]) / buy_n if buy_n > 0 else 0
        sell_n = len(result["sell_results"])
        sell_wins = sum(1 for r in result["sell_results"] if r < 0)
        print(f"  買い: {buy_wins}/{buy_n}的中 平均{buy_avg:+.2f}% | 売り: {sell_wins}/{sell_n}的中")

    # ===== 総合分析 =====
    print("\n" + "=" * 60)
    print("総合分析レポート")
    print("=" * 60)

    all_buy = [r for res in all_results for r in res["buy_results"]]
    all_sell = [r for res in all_results for r in res["sell_results"]]
    all_details = [d for res in all_results for d in res["score_details"]]

    discord_lines = ["**大規模バックテスト結果 (5年間×20期間)**\n"]

    if all_buy:
        wins = sum(1 for r in all_buy if r > 0)
        losses = [r for r in all_buy if r <= 0]
        gains = [r for r in all_buy if r > 0]
        avg = sum(all_buy) / len(all_buy)

        # シャープレシオ（リスクフリーレート0想定、年率換算なし）
        if len(all_buy) >= 2:
            mean_ret = sum(all_buy) / len(all_buy)
            std_ret = (sum((r - mean_ret) ** 2 for r in all_buy) / (len(all_buy) - 1)) ** 0.5
            sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        # 最大ドローダウン（累積リターンベース）
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in all_buy:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # プロフィットファクター（総利益 / 総損失）
        total_gain = sum(gains) if gains else 0
        total_loss = abs(sum(losses)) if losses else 0
        profit_factor = total_gain / total_loss if total_loss > 0 else float("inf")

        print(f"\n買い候補 全体:")
        print(f"  トレード数: {len(all_buy)}")
        print(f"  的中率: {wins}/{len(all_buy)} ({wins/len(all_buy)*100:.1f}%)")
        print(f"  平均騰落率: {avg:+.2f}%")
        print(f"  最大利益: {max(all_buy):+.2f}%")
        print(f"  最大損失: {min(all_buy):+.2f}%")
        print(f"  シャープレシオ: {sharpe:.2f}")
        print(f"  最大ドローダウン: {max_dd:.2f}%")
        print(f"  プロフィットファクター: {profit_factor:.2f}")

        discord_lines.append(
            f"買い: {wins}/{len(all_buy)}的中 ({wins/len(all_buy)*100:.0f}%) "
            f"平均{avg:+.2f}% 最大{max(all_buy):+.1f}%/{min(all_buy):+.1f}%"
        )
        discord_lines.append(
            f"Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2f}% | PF: {profit_factor:.2f}"
        )

    if all_sell:
        wins = sum(1 for r in all_sell if r < 0)
        avg = sum(all_sell) / len(all_sell)
        print(f"\n売り候補 全体:")
        print(f"  トレード数: {len(all_sell)}")
        print(f"  的中率: {wins}/{len(all_sell)} ({wins/len(all_sell)*100:.1f}%)")
        print(f"  平均騰落率: {avg:+.2f}%")
        discord_lines.append(
            f"売り: {wins}/{len(all_sell)}的中 ({wins/len(all_sell)*100:.0f}%) 平均{avg:+.2f}%"
        )

    # スコア内訳分析
    buy_details = [d for d in all_details if d["direction"] == "buy"]
    if buy_details:
        df_s = pd.DataFrame(buy_details)

        print(f"\nスコア内訳分析（買い {len(buy_details)}件）:")
        print("-" * 60)
        discord_lines.append(f"\n**スコア内訳分析 ({len(buy_details)}件)**")

        factors = ["trend", "macd", "volume", "fundamental", "rsi", "ichimoku", "pattern", "risk_reward", "sector"]
        for factor in factors:
            if factor not in df_s.columns:
                continue
            median = df_s[factor].median()
            high = df_s[df_s[factor] >= median]
            low = df_s[df_s[factor] < median]
            h_hit = high["hit"].mean() * 100 if len(high) > 0 else 0
            l_hit = low["hit"].mean() * 100 if len(low) > 0 else 0
            h_avg = high["pct"].mean() if len(high) > 0 else 0
            l_avg = low["pct"].mean() if len(low) > 0 else 0
            diff = h_hit - l_hit
            mark = "+++" if diff > 15 else "++" if diff > 5 else "+" if diff > 0 else "-" if diff > -5 else "--"

            line = (
                f"  {factor:12s}: 高群 {h_hit:.0f}%({h_avg:+.1f}%) "
                f"/ 低群 {l_hit:.0f}%({l_avg:+.1f}%) 差{diff:+.0f}% [{mark}]"
            )
            print(line)
            discord_lines.append(f"{factor}: 高{h_hit:.0f}%/低{l_hit:.0f}% [{mark}]")

        # スコア帯別的中率
        print(f"\nスコア帯別:")
        discord_lines.append(f"\n**スコア帯別的中率**")
        for lo, hi in [(55, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 100)]:
            band = df_s[(df_s["total"] >= lo) & (df_s["total"] < hi)]
            if len(band) > 0:
                hit = band["hit"].mean() * 100
                avg_r = band["pct"].mean()
                line = f"  {lo}-{hi}: {len(band)}件 的中{hit:.0f}% 平均{avg_r:+.2f}%"
                print(line)
                discord_lines.append(f"{lo}-{hi}: {len(band)}件 {hit:.0f}% {avg_r:+.2f}%")

    # セクター別分析
    if buy_details:
        df_s = pd.DataFrame(buy_details)
        if "sector_name" in df_s.columns:
            sector_groups = df_s.groupby("sector_name")
            print(f"\nセクター別的中率（買い）:")
            discord_lines.append(f"\n**セクター別的中率**")
            for sector_name, group in sorted(
                sector_groups, key=lambda x: len(x[1]), reverse=True
            ):
                if len(group) < 2 or not sector_name:
                    continue
                hit = group["hit"].mean() * 100
                avg_r = group["pct"].mean()
                line = f"  {sector_name}: {len(group)}件 的中{hit:.0f}% 平均{avg_r:+.2f}%"
                print(line)
                discord_lines.append(f"{sector_name}: {len(group)}件 {hit:.0f}% {avg_r:+.2f}%")

    # 期間別
    print(f"\n期間別:")
    for r in all_results:
        d = r["cutoff_date"]
        bn = len(r["buy_results"])
        bw = sum(1 for x in r["buy_results"] if x > 0)
        ba = sum(r["buy_results"]) / bn if bn > 0 else 0
        hr = bw / bn * 100 if bn > 0 else 0
        print(f"  {d}: {bw}/{bn} ({hr:.0f}%) {ba:+.2f}%")

    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.0f}秒")
    discord_lines.append(f"\n実行時間: {elapsed:.0f}秒")

    send_discord("\n".join(discord_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
