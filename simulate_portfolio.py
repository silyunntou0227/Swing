"""ポートフォリオシミュレーション: 100万円 × 10日付 × Top3

各日付でスコア上位3銘柄に均等配分で投資。
100株単位で購入。保有中にSL/TP到達で即決済、未達なら推奨保有日数後に決済。
累積損益を追跡する。
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
from src.scoring.risk import RiskCalculator
from src.scoring.scorer import MultiFactorScorer
from src.screening.pipeline import ScreeningPipeline
from src.utils.logging_config import logger

INITIAL_CAPITAL = 1_000_000  # 初期資金100万円
TOP_N = 3  # 各日付の購入銘柄数
RANDOM_SEED = 99  # 日付選択のシード（前回42、今回99で別日付）


def generate_random_dates(n: int = 10, seed: int = RANDOM_SEED) -> list[date]:
    """過去5年間からランダムに営業日をn個選択（固定シード、30日以上間隔）"""
    end = date.today() - timedelta(days=10)
    start = date(end.year - 5, end.month, end.day)
    weekdays = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            weekdays.append(current)
        current += timedelta(days=1)

    random.seed(seed)
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
                chunk, period="5y", auto_adjust=True, threads=True, progress=False,
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
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True).dt.tz_localize(None)
    prices["Code"] = prices["Code"].astype(str).str[:4]
    prices = prices.sort_values(["Code", "Date"]).reset_index(drop=True)
    print(f"  完了: {prices['Code'].nunique()}銘柄, {len(prices)}行")
    return prices


def simulate_single_date(
    cutoff_date: date,
    all_prices: pd.DataFrame,
    stocks: pd.DataFrame,
    capital: float,
) -> tuple[list[dict], float]:
    """単一日付でスクリーニング→スコアリング→ポジション構築→損益計算

    Returns:
        (trades, new_capital): 各トレード詳細と、決済後の資金
    """
    trades = []

    cutoff_dt = pd.Timestamp(cutoff_date)
    start_dt = cutoff_dt - pd.Timedelta(days=400)
    prices_before = all_prices[
        (all_prices["Date"] >= start_dt) & (all_prices["Date"] <= cutoff_dt)
    ].copy()

    if prices_before.empty or prices_before["Code"].nunique() < 50:
        print(f"    データ不足 — スキップ")
        return trades, capital

    market_data = MarketData(
        stocks=stocks, prices=prices_before,
        financials=pd.DataFrame(), scan_date=cutoff_date,
    )

    # スクリーニング
    try:
        pipeline = ScreeningPipeline()
        candidates = pipeline.run(market_data)
    except Exception as e:
        print(f"    スクリーニングエラー: {e}")
        return trades, capital

    # スコアリング
    try:
        scorer = MultiFactorScorer()
        scored_buy = scorer.score(candidates.buy, market_data, direction="buy")
    except Exception as e:
        print(f"    スコアリングエラー: {e}")
        return trades, capital

    # リスク計算
    risk_calc = RiskCalculator(capital=capital)
    for c in scored_buy:
        risk_calc.calculate(c, market_data)

    top_buy = sorted(scored_buy, key=lambda x: x.total_score, reverse=True)[:TOP_N]

    if not top_buy:
        print(f"    買い候補なし — スキップ")
        return trades, capital

    # 均等配分（資金をTOP_N等分）
    alloc_per_stock = capital / TOP_N
    remaining_capital = capital

    for c in top_buy:
        entry_data = prices_before[prices_before["Code"] == c.code]
        if entry_data.empty:
            continue
        entry_price = float(entry_data.iloc[-1]["Close"])
        entry_date = entry_data.iloc[-1]["Date"]

        if entry_price <= 0:
            continue

        # 100株単位で購入
        max_shares = int(alloc_per_stock / entry_price)
        shares = max(100, (max_shares // 100) * 100)
        if shares * entry_price > alloc_per_stock * 1.05:
            # 予算オーバー → 100株に制限
            shares = 100
        if shares * entry_price > remaining_capital:
            # 資金不足
            shares = (int(remaining_capital / entry_price) // 100) * 100
            if shares < 100:
                continue

        cost = shares * entry_price
        remaining_capital -= cost

        # 売却: 保有中に日足High/LowでSL/TP判定 → 未達なら推奨日数後に決済
        hold_days = c.recommended_hold_days if c.recommended_hold_days > 0 else 5
        sl = c.stop_loss
        tp = c.take_profit
        after_data = all_prices[
            (all_prices["Code"] == c.code)
            & (all_prices["Date"] > cutoff_dt)
        ].head(hold_days)

        exit_price = entry_price
        exit_date_str = "N/A"
        actual_hold = 0
        exit_reason = "データなし"

        if not after_data.empty:
            exited = False
            for day_idx, (_, row) in enumerate(after_data.iterrows(), 1):
                day_low = float(row["Low"]) if "Low" in row and pd.notna(row["Low"]) else None
                day_high = float(row["High"]) if "High" in row and pd.notna(row["High"]) else None

                # 損切り判定: 安値がSLを下回ったら損切り（SL価格で約定と仮定）
                if sl > 0 and day_low is not None and day_low <= sl:
                    exit_price = sl
                    exit_date_str = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
                    actual_hold = day_idx
                    exit_reason = "SL損切り"
                    exited = True
                    break

                # 利確判定: 高値がTPを上回ったら利確（TP価格で約定と仮定）
                if tp > 0 and day_high is not None and day_high >= tp:
                    exit_price = tp
                    exit_date_str = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
                    actual_hold = day_idx
                    exit_reason = "TP利確"
                    exited = True
                    break

            # SL/TP未達 → 推奨保有日数後の終値で決済
            if not exited:
                exit_price = float(after_data.iloc[-1]["Close"])
                exit_date_str = pd.Timestamp(after_data.iloc[-1]["Date"]).strftime("%Y-%m-%d")
                actual_hold = len(after_data)
                exit_reason = "期限決済"

        proceeds = shares * exit_price
        pnl = proceeds - cost
        pnl_pct = (exit_price - entry_price) / entry_price * 100

        remaining_capital += proceeds

        trades.append({
            "code": c.code,
            "name": c.name,
            "score": round(c.total_score, 1),
            "entry_date": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 1),
            "exit_date": exit_date_str,
            "exit_price": round(exit_price, 1),
            "shares": shares,
            "cost": round(cost),
            "proceeds": round(proceeds),
            "pnl": round(pnl),
            "pnl_pct": round(pnl_pct, 2),
            "hold_days": actual_hold,
            "stop_loss": round(sl, 1),
            "take_profit": round(tp, 1),
            "exit_reason": exit_reason,
            "hit": pnl > 0,
        })

    # 未使用資金を戻す
    new_capital = remaining_capital
    return trades, new_capital


def main():
    t0 = time.time()

    print("=" * 70)
    print(f"ポートフォリオシミュレーション: 初期資金{INITIAL_CAPITAL:,}円 × 10日付 × Top3")
    print("=" * 70)

    # ===== フェーズ1: データ準備 =====
    print(f"\n{'━' * 70}")
    print("フェーズ1: データ準備")
    print(f"{'━' * 70}")

    print("\n[1/2] 銘柄一覧取得...")
    stocks = fetch_jpx_stock_list()
    codes = get_tradeable_codes(stocks)
    print(f"  対象: {len(codes)}銘柄")

    print(f"\n[2/2] 5年間の株価データを一括ダウンロード...")
    all_prices = download_all_prices(codes)
    if all_prices.empty:
        print("ERROR: 株価データ取得失敗")
        return 1
    print(f"  データ準備完了: {time.time()-t0:.0f}秒")

    dates = generate_random_dates(10)

    # ===== フェーズ2: 各日付でスクリーニング→売買 =====
    print(f"\n{'━' * 70}")
    print("フェーズ2: 日付別売買シミュレーション")
    print(f"{'━' * 70}")

    capital = INITIAL_CAPITAL
    all_trades = []
    period_summaries = []

    for i, cutoff in enumerate(dates):
        elapsed = time.time() - t0
        print(f"\n{'─' * 70}")
        print(f"期間 {i+1}/10: {cutoff} ({cutoff.strftime('%a')}) | 資金: {capital:,.0f}円 [{elapsed:.0f}s]")
        print(f"{'─' * 70}")

        capital_before = capital
        trades, capital = simulate_single_date(cutoff, all_prices, stocks, capital)

        if trades:
            period_pnl = sum(t["pnl"] for t in trades)
            period_pnl_pct = (capital - capital_before) / capital_before * 100
            wins = sum(1 for t in trades if t["hit"])

            print(f"\n  {'No':>2} {'銘柄':12s} {'Score':>5} {'買値':>8} {'売値':>8} "
                  f"{'株数':>5} {'損益':>10} {'損益%':>7} {'保有':>3} {'決済理由':8s} {'判定':>3}")
            print(f"  {'──':>2} {'────────────':12s} {'─────':>5} {'────────':>8} {'────────':>8} "
                  f"{'─────':>5} {'──────────':>10} {'───────':>7} {'───':>3} {'────────':8s} {'───':>3}")

            for j, t in enumerate(trades, 1):
                mark = "○" if t["hit"] else "×"
                print(
                    f"  {j:2d} {t['code']} {t['name'][:6]:6s} "
                    f"{t['score']:5.1f} "
                    f"{t['entry_price']:8,.0f} {t['exit_price']:8,.0f} "
                    f"{t['shares']:5d} "
                    f"{t['pnl']:+10,.0f} "
                    f"{t['pnl_pct']:+6.2f}% "
                    f"{t['hold_days']:3d}d "
                    f" {t['exit_reason']:8s}"
                    f" {mark}"
                )

            print(f"\n  期間損益: {period_pnl:+,.0f}円 ({period_pnl_pct:+.2f}%) | "
                  f"的中: {wins}/{len(trades)} | 決済後資金: {capital:,.0f}円")

            period_summaries.append({
                "date": cutoff.isoformat(),
                "capital_before": capital_before,
                "capital_after": capital,
                "pnl": period_pnl,
                "pnl_pct": round(period_pnl_pct, 2),
                "trades": len(trades),
                "wins": wins,
            })
            all_trades.extend(trades)
        else:
            print(f"  → トレードなし（資金維持: {capital:,.0f}円）")
            period_summaries.append({
                "date": cutoff.isoformat(),
                "capital_before": capital_before,
                "capital_after": capital,
                "pnl": 0,
                "pnl_pct": 0,
                "trades": 0,
                "wins": 0,
            })

    # ===== フェーズ3: 総合結果 =====
    print(f"\n{'━' * 70}")
    print("フェーズ3: 総合結果")
    print(f"{'━' * 70}")

    total_pnl = capital - INITIAL_CAPITAL
    total_pnl_pct = total_pnl / INITIAL_CAPITAL * 100
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t["hit"])

    print(f"\n┌─────────────────────────────────────────────┐")
    print(f"│  初期資金:       {INITIAL_CAPITAL:>12,}円              │")
    print(f"│  最終資金:       {capital:>12,.0f}円              │")
    print(f"│  総損益:         {total_pnl:>+12,.0f}円 ({total_pnl_pct:+.2f}%)   │")
    print(f"│  トレード数:     {total_trades:>12}回              │")
    if total_trades > 0:
        print(f"│  的中率:         {total_wins}/{total_trades} ({total_wins/total_trades*100:.1f}%)               │")
        avg_pnl = sum(t["pnl"] for t in all_trades) / total_trades
        print(f"│  平均損益/トレード: {avg_pnl:>+10,.0f}円              │")
        max_win = max(t["pnl"] for t in all_trades)
        max_loss = min(t["pnl"] for t in all_trades)
        print(f"│  最大利益:       {max_win:>+12,}円              │")
        print(f"│  最大損失:       {max_loss:>+12,}円              │")
    print(f"└─────────────────────────────────────────────┘")

    # 期間別推移
    print(f"\n期間別資金推移:")
    print(f"  {'日付':12s} {'資金(前)':>12s} {'損益':>10s} {'損益%':>7s} {'的中':>5s} {'資金(後)':>12s}")
    print(f"  {'────────────':12s} {'────────────':>12s} {'──────────':>10s} {'───────':>7s} {'─────':>5s} {'────────────':>12s}")
    for s in period_summaries:
        wins_str = f"{s['wins']}/{s['trades']}" if s['trades'] > 0 else "---"
        print(
            f"  {s['date']:12s} {s['capital_before']:>12,.0f} "
            f"{s['pnl']:>+10,.0f} {s['pnl_pct']:>+6.2f}% "
            f"{wins_str:>5s} {s['capital_after']:>12,.0f}"
        )

    # 決済理由別集計
    if all_trades:
        sl_trades = [t for t in all_trades if t["exit_reason"] == "SL損切り"]
        tp_trades = [t for t in all_trades if t["exit_reason"] == "TP利確"]
        time_trades = [t for t in all_trades if t["exit_reason"] == "期限決済"]
        print(f"\n決済理由別集計:")
        print(f"  SL損切り: {len(sl_trades)}回", end="")
        if sl_trades:
            sl_avg = sum(t["pnl_pct"] for t in sl_trades) / len(sl_trades)
            print(f" | 平均{sl_avg:+.2f}%")
        else:
            print()
        print(f"  TP利確:   {len(tp_trades)}回", end="")
        if tp_trades:
            tp_avg = sum(t["pnl_pct"] for t in tp_trades) / len(tp_trades)
            print(f" | 平均{tp_avg:+.2f}%")
        else:
            print()
        print(f"  期限決済: {len(time_trades)}回", end="")
        if time_trades:
            time_avg = sum(t["pnl_pct"] for t in time_trades) / len(time_trades)
            time_wins = sum(1 for t in time_trades if t["hit"])
            print(f" | 平均{time_avg:+.2f}% | 的中{time_wins}/{len(time_trades)}")
        else:
            print()

    # 全トレード一覧
    if all_trades:
        print(f"\n全トレード一覧:")
        print(f"  {'No':>3} {'日付':10s} {'銘柄':12s} {'買値':>8} {'売値':>8} "
              f"{'株数':>5} {'損益':>10} {'損益%':>7} {'Hold':>4} {'決済':8s} {'判定':>3}")
        print(f"  {'───':>3} {'──────────':10s} {'────────────':12s} {'────────':>8} {'────────':>8} "
              f"{'─────':>5} {'──────────':>10} {'───────':>7} {'────':>4} {'────────':8s} {'───':>3}")
        for j, t in enumerate(all_trades, 1):
            mark = "○" if t["hit"] else "×"
            print(
                f"  {j:3d} {t['entry_date']:10s} "
                f"{t['code']} {t['name'][:6]:6s} "
                f"{t['entry_price']:8,.0f} {t['exit_price']:8,.0f} "
                f"{t['shares']:5d} "
                f"{t['pnl']:+10,} "
                f"{t['pnl_pct']:+6.2f}% "
                f"{t['hold_days']:4d}d "
                f" {t['exit_reason']:8s}"
                f" {mark}"
            )

    elapsed = time.time() - t0
    print(f"\n総実行時間: {elapsed:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
