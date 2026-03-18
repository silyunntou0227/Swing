"""3日前バックテスト: スクリーニング精度検証

スクリーニング日（cutoff）時点で分析を実行し、
検証日（check）の実際の値動きと比較する。
結果をDiscordに通知する。
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timedelta

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
    """Discord Webhook にテキストを送信"""
    if not DISCORD_WEBHOOK_URL:
        return
    import requests
    try:
        # 2000文字制限対策: 分割送信
        for i in range(0, len(content), 1900):
            chunk = content[i:i+1900]
            requests.post(
                DISCORD_WEBHOOK_URL,
                json={"content": chunk},
                timeout=10,
            )
            time.sleep(0.5)
    except Exception as e:
        print(f"Discord送信エラー: {e}")


def main():
    # 環境変数またはデフォルト値
    cutoff_str = os.environ.get("CUTOFF_DATE", "2026-03-13")
    check_str = os.environ.get("CHECK_DATE", "2026-03-18")
    CUTOFF_DATE = date.fromisoformat(cutoff_str)
    CHECK_DATE = date.fromisoformat(check_str)

    header = (
        f"{'=' * 50}\n"
        f"バックテスト: {CUTOFF_DATE} 時点のスクリーニング\n"
        f"検証期間: {CUTOFF_DATE} → {CHECK_DATE}\n"
        f"{'=' * 50}"
    )
    print(header)

    # Step 1: 銘柄一覧取得
    print("\n[1/5] 銘柄一覧取得...")
    stocks = fetch_jpx_stock_list()
    codes = get_tradeable_codes(stocks, max_stocks=1200)
    print(f"  対象: {len(codes)}銘柄")

    if not codes:
        print("ERROR: 銘柄リスト取得失敗")
        return 1

    # Step 2: 株価データ取得（1年分 → cutoff以前に絞る）
    print("\n[2/5] 株価データ取得...")
    yahoo = YahooClient()
    prices = yahoo.fetch_bulk_prices(codes, period="1y")

    if prices.empty:
        print("ERROR: 株価データ取得失敗")
        return 1

    # 前処理
    if "Date" in prices.columns:
        prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    if "Code" in prices.columns:
        prices["Code"] = prices["Code"].astype(str).str[:4]

    cutoff_dt = pd.Timestamp(CUTOFF_DATE)
    check_dt = pd.Timestamp(CHECK_DATE)

    # cutoff以前のデータのみでスクリーニング
    prices_before = prices[prices["Date"] <= cutoff_dt].copy()
    prices_before = prices_before.sort_values(["Code", "Date"]).reset_index(drop=True)

    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing = [c for c in price_cols if c in prices_before.columns]
    if existing:
        prices_before = prices_before.dropna(subset=existing)

    n_stocks = prices_before["Code"].nunique()
    print(f"  {n_stocks}銘柄, {len(prices_before)}行（〜{CUTOFF_DATE}）")

    # Step 3: スクリーニング
    print("\n[3/5] スクリーニング実行...")
    market_data = MarketData(
        stocks=stocks,
        prices=prices_before,
        financials=pd.DataFrame(),
        scan_date=CUTOFF_DATE,
    )

    pipeline = ScreeningPipeline()
    candidates = pipeline.run(market_data)
    print(f"  買い候補: {len(candidates.buy)}件, 売り候補: {len(candidates.sell)}件")

    # Step 4: スコアリング
    print("\n[4/5] スコアリング...")
    scorer = MultiFactorScorer()
    scored_buy = scorer.score(candidates.buy, market_data, direction="buy")
    scored_sell = scorer.score(candidates.sell, market_data, direction="sell")

    top_buy = sorted(scored_buy, key=lambda x: x.total_score, reverse=True)[:TOP_BUY_CANDIDATES]
    top_sell = sorted(scored_sell, key=lambda x: x.total_score, reverse=True)[:TOP_SELL_CANDIDATES]

    if not top_buy and not top_sell:
        msg = "バックテスト: 候補銘柄なし（フィルタが厳しすぎる可能性）"
        print(msg)
        send_discord(msg)
        return 0

    # Step 5: 実際の値動きと比較
    print("\n[5/5] 検証...")

    # cutoff後〜check日のデータ
    prices_after = prices[
        (prices["Date"] > cutoff_dt) & (prices["Date"] <= check_dt)
    ].copy()

    output_lines = []
    output_lines.append(f"**バックテスト結果 ({CUTOFF_DATE} → {CHECK_DATE})**\n")

    # --- 買い候補 ---
    output_lines.append(f"**買い候補 Top{len(top_buy)}**")
    buy_results = []

    for i, c in enumerate(top_buy, 1):
        code = c.code
        entry_data = prices_before[prices_before["Code"] == code]
        if entry_data.empty:
            continue
        entry_price = float(entry_data.iloc[-1]["Close"])

        # 検証日の終値
        exit_price = _get_exit_price(code, prices_after, CHECK_DATE)

        if exit_price is not None:
            pct = (exit_price - entry_price) / entry_price * 100
            mark = "O" if pct > 0 else "X"
            buy_results.append(pct)
            price_str = f"¥{entry_price:,.0f}→¥{exit_price:,.0f} ({pct:+.2f}%) {mark}"
        else:
            price_str = f"¥{entry_price:,.0f}→? (N/A)"

        signals_short = ", ".join(c.signals[:2]) if c.signals else "-"
        sector_info = f" [{c.sector_name}]" if c.sector_name else ""
        line = f"#{i} {c.name}({code}){sector_info} スコア{c.total_score:.0f} | {price_str}"
        output_lines.append(line)
        print(f"  {line}")
        print(f"      シグナル: {signals_short}")
        if c.sector_explanation:
            print(f"      セクター: {c.sector_explanation}")

    # --- 売り候補 ---
    output_lines.append(f"\n**売り候補 Top{len(top_sell)}**")
    sell_results = []

    for i, c in enumerate(top_sell, 1):
        code = c.code
        entry_data = prices_before[prices_before["Code"] == code]
        if entry_data.empty:
            continue
        entry_price = float(entry_data.iloc[-1]["Close"])

        exit_price = _get_exit_price(code, prices_after, CHECK_DATE)

        if exit_price is not None:
            pct = (exit_price - entry_price) / entry_price * 100
            mark = "O" if pct < 0 else "X"  # 売りは下落が正解
            sell_results.append(pct)
            price_str = f"¥{entry_price:,.0f}→¥{exit_price:,.0f} ({pct:+.2f}%) {mark}"
        else:
            price_str = f"¥{entry_price:,.0f}→? (N/A)"

        sector_info = f" [{c.sector_name}]" if c.sector_name else ""
        line = f"#{i} {c.name}({code}){sector_info} スコア{c.total_score:.0f} | {price_str}"
        output_lines.append(line)
        print(f"  {line}")

    # --- サマリー ---
    output_lines.append(f"\n**検証サマリー**")

    if buy_results:
        buy_wins = sum(1 for r in buy_results if r > 0)
        buy_avg = sum(buy_results) / len(buy_results)
        line = (f"買い: {buy_wins}/{len(buy_results)}的中 "
                f"({buy_wins/len(buy_results)*100:.0f}%) "
                f"平均{buy_avg:+.2f}%")
        output_lines.append(line)
        print(f"  {line}")

    if sell_results:
        sell_wins = sum(1 for r in sell_results if r < 0)
        sell_avg = sum(sell_results) / len(sell_results)
        line = (f"売り: {sell_wins}/{len(sell_results)}的中 "
                f"({sell_wins/len(sell_results)*100:.0f}%) "
                f"平均{sell_avg:+.2f}%")
        output_lines.append(line)
        print(f"  {line}")

    # 日経平均との比較
    try:
        nk = yf.Ticker("^N225")
        nk_hist = nk.history(start=CUTOFF_DATE.isoformat(), end=(CHECK_DATE + timedelta(days=1)).isoformat())
        if len(nk_hist) >= 2:
            nk_start = float(nk_hist["Close"].iloc[0])
            nk_end = float(nk_hist["Close"].iloc[-1])
            nk_pct = (nk_end - nk_start) / nk_start * 100
            nk_line = f"日経平均: ¥{nk_start:,.0f}→¥{nk_end:,.0f} ({nk_pct:+.2f}%)"
            output_lines.append(nk_line)
            print(f"  {nk_line}")
            if buy_results:
                alpha = buy_avg - nk_pct
                alpha_line = f"買い候補の超過リターン(α): {alpha:+.2f}%"
                output_lines.append(alpha_line)
                print(f"  {alpha_line}")
    except Exception as e:
        print(f"  日経平均取得エラー: {e}")

    # Discord送信
    discord_msg = "\n".join(output_lines)
    send_discord(discord_msg)
    print("\nDiscord通知送信完了")

    return 0


def _get_exit_price(code: str, prices_after: pd.DataFrame, check_date: date) -> float | None:
    """検証日の終値を取得（バルクデータ → 個別フォールバック）"""
    if not prices_after.empty:
        exit_data = prices_after[prices_after["Code"] == code]
        if not exit_data.empty:
            return float(exit_data.iloc[-1]["Close"])

    # フォールバック: 個別取得
    try:
        t = yf.Ticker(f"{code}.T")
        hist = t.history(
            start=(check_date - timedelta(days=5)).isoformat(),
            end=(check_date + timedelta(days=1)).isoformat(),
        )
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass

    return None


if __name__ == "__main__":
    sys.exit(main() or 0)
