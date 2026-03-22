"""10日付バックテスト: ランダムな過去10日付でスクリーニング→スコアリング→リスク計算→Top3出力

backtest_multi.py をベースに、RiskCalculator も実行し、
各日付のTop3銘柄（買い/売り）の詳細を出力する。
"""
from __future__ import annotations

import random
import sys
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from src.data.data_loader import MarketData
from src.data.stock_list import fetch_jpx_stock_list, get_tradeable_codes
from src.config import TOP_BUY_CANDIDATES, TOP_SELL_CANDIDATES
from src.scoring.risk import RiskCalculator
from src.scoring.scorer import MultiFactorScorer
from src.screening.pipeline import ScreeningPipeline
from src.utils.logging_config import logger


TOP_N = 3  # 各日付で表示する上位銘柄数


def generate_random_dates(n: int = 10) -> list[date]:
    """過去5年間からランダムに営業日をn個選択（固定シード、30日以上間隔）"""
    end = date.today() - timedelta(days=10)
    start = date(end.year - 5, end.month, end.day)
    weekdays = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            weekdays.append(current)
        current += timedelta(days=1)

    random.seed(42)
    shuffled = weekdays.copy()
    random.shuffle(shuffled)

    selected = []
    for d in shuffled:
        if len(selected) >= n:
            break
        if all(abs((d - s).days) >= 30 for s in selected):
            selected.append(d)
    return sorted(selected)


def download_all_prices(codes: list[str]) -> pd.DataFrame:
    """全銘柄の5年間株価を一括ダウンロード"""
    tickers = [f"{c}.T" for c in codes]
    chunk_size = 200
    all_frames = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(tickers) - 1) // chunk_size + 1
        print(f"  DL: チャンク {chunk_num}/{total_chunks} ({len(chunk)}銘柄)")

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

            if isinstance(data.columns, pd.MultiIndex):
                for ticker in chunk:
                    code = ticker.replace(".T", "")
                    try:
                        df_single = data.xs(ticker, level=1, axis=1).copy()
                        df_single = df_single.reset_index()
                        if "Date" not in df_single.columns and "index" in df_single.columns:
                            df_single = df_single.rename(columns={"index": "Date"})
                        if "Date" not in df_single.columns:
                            df_single = df_single.reset_index()
                            for col in df_single.columns:
                                if "date" in str(col).lower():
                                    df_single = df_single.rename(columns={col: "Date"})
                                    break
                        df_single["Code"] = code
                        if not df_single.empty and "Close" in df_single.columns:
                            cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "Code"] if c in df_single.columns]
                            all_frames.append(df_single[cols].dropna(subset=["Close"]))
                    except (KeyError, TypeError):
                        continue
            else:
                data = data.reset_index()
                if len(chunk) == 1:
                    data["Code"] = chunk[0].replace(".T", "")
                    if "Close" in data.columns:
                        cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "Code"] if c in data.columns]
                        all_frames.append(data[cols].dropna(subset=["Close"]))
        except Exception as e:
            print(f"    チャンクエラー: {e}")
            continue

        time.sleep(1)

    if not all_frames:
        return pd.DataFrame()

    prices = pd.concat(all_frames, ignore_index=True)
    # TZ正規化: TZ-aware → naive に統一
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    prices["Code"] = prices["Code"].astype(str).str[:4]
    prices = prices.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"  完了: {prices['Code'].nunique()}銘柄, {len(prices)}行")
    return prices


def run_single_date(
    cutoff_date: date,
    all_prices: pd.DataFrame,
    stocks: pd.DataFrame,
    check_days: int = 7,
) -> dict:
    """単一日付でスクリーニング→スコアリング→リスク計算→検証"""
    result = {
        "cutoff_date": cutoff_date.isoformat(),
        "buy_candidates": [],
        "sell_candidates": [],
        "buy_results": [],
        "sell_results": [],
    }

    cutoff_dt = pd.Timestamp(cutoff_date)
    start_dt = cutoff_dt - pd.Timedelta(days=400)
    prices_before = all_prices[
        (all_prices["Date"] >= start_dt) & (all_prices["Date"] <= cutoff_dt)
    ].copy()

    if prices_before.empty or prices_before["Code"].nunique() < 50:
        return result

    end_dt = cutoff_dt + pd.Timedelta(days=check_days * 2)
    prices_after = all_prices[
        (all_prices["Date"] > cutoff_dt) & (all_prices["Date"] <= end_dt)
    ].copy()

    market_data = MarketData(
        stocks=stocks,
        prices=prices_before,
        financials=pd.DataFrame(),
        scan_date=cutoff_date,
    )

    # スクリーニング
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

    # リスク計算
    risk_calc = RiskCalculator()
    for c in scored_buy:
        risk_calc.calculate(c, market_data)
    for c in scored_sell:
        risk_calc.calculate(c, market_data)

    top_buy = sorted(scored_buy, key=lambda x: x.total_score, reverse=True)[:TOP_N]
    top_sell = sorted(scored_sell, key=lambda x: x.total_score, reverse=True)[:TOP_N]

    # 買いTop3の検証
    for c in top_buy:
        entry_data = prices_before[prices_before["Code"] == c.code]
        if entry_data.empty:
            continue
        entry_price = float(entry_data.iloc[-1]["Close"])
        hold = c.recommended_hold_days if c.recommended_hold_days > 0 else check_days
        after_data = prices_after[prices_after["Code"] == c.code].head(hold)

        pct = None
        if not after_data.empty:
            exit_price = float(after_data.iloc[-1]["Close"])
            pct = (exit_price - entry_price) / entry_price * 100
            result["buy_results"].append(pct)

        result["buy_candidates"].append({
            "code": c.code,
            "name": c.name,
            "score": round(c.total_score, 1),
            "close": round(entry_price, 1),
            "stop_loss": round(c.stop_loss, 1),
            "take_profit": round(c.take_profit, 1),
            "rr_ratio": round(c.risk_reward_ratio, 1),
            "hold_days": hold,
            "pct": round(pct, 2) if pct is not None else None,
            "hit": pct > 0 if pct is not None else None,
        })

    # 売りTop3の検証
    for c in top_sell:
        entry_data = prices_before[prices_before["Code"] == c.code]
        if entry_data.empty:
            continue
        entry_price = float(entry_data.iloc[-1]["Close"])
        hold = c.recommended_hold_days if c.recommended_hold_days > 0 else check_days
        after_data = prices_after[prices_after["Code"] == c.code].head(hold)

        pct = None
        if not after_data.empty:
            exit_price = float(after_data.iloc[-1]["Close"])
            pct = (exit_price - entry_price) / entry_price * 100
            result["sell_results"].append(pct)

        result["sell_candidates"].append({
            "code": c.code,
            "name": c.name,
            "score": round(c.total_score, 1),
            "close": round(entry_price, 1),
            "stop_loss": round(c.stop_loss, 1),
            "take_profit": round(c.take_profit, 1),
            "rr_ratio": round(c.risk_reward_ratio, 1),
            "hold_days": hold,
            "pct": round(pct, 2) if pct is not None else None,
            "hit": (pct < 0) if pct is not None else None,
        })

    return result


def print_date_result(result: dict) -> None:
    """単一日付の結果を見やすく表示"""
    d = result["cutoff_date"]

    if result["buy_candidates"]:
        print(f"  【買いTop{len(result['buy_candidates'])}】")
        for i, c in enumerate(result["buy_candidates"], 1):
            hit_mark = ""
            if c["pct"] is not None:
                hit_mark = f" → {c['pct']:+.2f}% {'○' if c['hit'] else '×'}"
            print(
                f"    {i}. {c['code']} {c['name'][:8]:8s} "
                f"Score={c['score']:5.1f} "
                f"@{c['close']:.0f} "
                f"SL={c['stop_loss']:.0f} TP={c['take_profit']:.0f} "
                f"RR={c['rr_ratio']:.1f} "
                f"Hold={c['hold_days']}d"
                f"{hit_mark}"
            )
    else:
        print(f"  【買い候補なし】")

    if result["sell_candidates"]:
        print(f"  【売りTop{len(result['sell_candidates'])}】")
        for i, c in enumerate(result["sell_candidates"], 1):
            hit_mark = ""
            if c["pct"] is not None:
                hit_mark = f" → {c['pct']:+.2f}% {'○' if c['hit'] else '×'}"
            print(
                f"    {i}. {c['code']} {c['name'][:8]:8s} "
                f"Score={c['score']:5.1f} "
                f"@{c['close']:.0f} "
                f"SL={c['stop_loss']:.0f} TP={c['take_profit']:.0f} "
                f"RR={c['rr_ratio']:.1f}"
                f"{hit_mark}"
            )


def main():
    t0 = time.time()

    print("=" * 60)
    print("日付別バックテスト: 10日付 × Top3")
    print("=" * 60)

    # 銘柄一覧
    print("\n[1/4] 銘柄一覧取得...")
    stocks = fetch_jpx_stock_list()
    codes = get_tradeable_codes(stocks)
    print(f"  対象: {len(codes)}銘柄")

    # 株価ダウンロード
    print(f"\n[2/4] 5年間の株価データを一括ダウンロード...")
    all_prices = download_all_prices(codes)
    if all_prices.empty:
        print("ERROR: 株価データ取得失敗")
        return 1
    print(f"  ダウンロード完了: {time.time()-t0:.0f}秒")

    # 日付生成
    dates = generate_random_dates(10)
    print(f"\n[3/4] テスト日付: {len(dates)}日")
    for i, d in enumerate(dates):
        print(f"  {i+1}. {d} ({d.strftime('%a')})")

    # 各日付でバックテスト
    print(f"\n[4/4] バックテスト実行中...")
    all_results = []

    for i, cutoff in enumerate(dates):
        elapsed = time.time() - t0
        print(f"\n{'─' * 50}")
        print(f"期間 {i+1}/{len(dates)}: {cutoff} ({cutoff.strftime('%a')}) [{elapsed:.0f}s]")

        result = run_single_date(cutoff, all_prices, stocks)
        all_results.append(result)
        print_date_result(result)

    # ===== 総合集計 =====
    print(f"\n{'=' * 60}")
    print("総合集計")
    print("=" * 60)

    all_buy_pcts = [r for res in all_results for r in res["buy_results"]]
    all_sell_pcts = [r for res in all_results for r in res["sell_results"]]

    if all_buy_pcts:
        wins = sum(1 for r in all_buy_pcts if r > 0)
        avg = sum(all_buy_pcts) / len(all_buy_pcts)
        print(f"\n買い候補:")
        print(f"  トレード数: {len(all_buy_pcts)}")
        print(f"  的中率: {wins}/{len(all_buy_pcts)} ({wins/len(all_buy_pcts)*100:.1f}%)")
        print(f"  平均騰落率: {avg:+.2f}%")
        print(f"  最大利益: {max(all_buy_pcts):+.2f}%")
        print(f"  最大損失: {min(all_buy_pcts):+.2f}%")
    else:
        print("\n買い候補: データなし")

    if all_sell_pcts:
        wins = sum(1 for r in all_sell_pcts if r < 0)
        avg = sum(all_sell_pcts) / len(all_sell_pcts)
        print(f"\n売り候補:")
        print(f"  トレード数: {len(all_sell_pcts)}")
        print(f"  的中率: {wins}/{len(all_sell_pcts)} ({wins/len(all_sell_pcts)*100:.1f}%)")
        print(f"  平均騰落率: {avg:+.2f}%")
    else:
        print("\n売り候補: データなし")

    # 期間別サマリ
    print(f"\n期間別サマリ:")
    for r in all_results:
        d = r["cutoff_date"]
        bn = len(r["buy_results"])
        bw = sum(1 for x in r["buy_results"] if x > 0)
        ba = sum(r["buy_results"]) / bn if bn > 0 else 0
        hr = bw / bn * 100 if bn > 0 else 0
        print(f"  {d}: 買い {bw}/{bn} ({hr:.0f}%) {ba:+.2f}%")

    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
