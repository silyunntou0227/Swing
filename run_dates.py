"""指定日付のおすすめ銘柄スクリーニング"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from src.screening.pipeline import ScreeningPipeline
from src.scoring.scorer import MultiFactorScorer
from src.scoring.risk import RiskCalculator
from src.data.data_loader import MarketData
from src.utils.logging_config import logger


# 主要日本株銘柄コード（時価総額上位 + 流動性の高い銘柄）
MAJOR_CODES = [
    # 大型株（日経225主要構成銘柄）
    "7203", "6758", "9984", "8306", "6861", "9433", "6902", "4063", "6501", "7267",
    "8035", "6098", "4502", "4503", "6954", "8058", "8031", "2914", "3382", "9432",
    "7741", "4661", "6367", "7974", "4568", "6981", "7751", "9983", "6273", "4519",
    "6857", "7752", "8801", "8802", "5401", "7269", "3407", "4901", "6702", "6752",
    "8411", "8316", "2502", "2802", "3086", "8766", "8725", "8267", "1925", "1928",
    "2413", "4452", "6301", "6503", "7201", "7211", "7261", "9020", "9021", "9022",
    "1802", "1803", "1812", "1801", "2801", "2871", "3101", "3401", "3402", "4005",
    "4021", "4042", "4183", "4188", "4208", "4272", "4307", "4324", "4543", "4578",
    "4689", "4704", "4755", "4911", "5108", "5201", "5202", "5301", "5332", "5333",
    "5411", "5713", "5714", "5801", "5802", "5803", "6103", "6113", "6178", "6201",
    "6326", "6361", "6471", "6504", "6506", "6594", "6645", "6674", "6701", "6724",
    "6753", "6762", "6770", "6841", "6869", "6920", "6952", "6971", "7004", "7011",
    "7012", "7013", "7186", "7205", "7270", "7272", "7731", "7733", "7735", "7762",
    "7832", "7911", "7912", "7951", "8001", "8002", "8015", "8053", "8252", "8253",
    "8309", "8331", "8354", "8355", "8473", "8591", "8601", "8604", "8628", "8630",
    "8697", "8750", "8795", "9001", "9005", "9007", "9008", "9064", "9101", "9104",
    "9107", "9201", "9202", "9301", "9434", "9501", "9502", "9503", "9531", "9532",
    "9602", "9613", "9735", "9766", "4528", "3659", "2432", "6988", "3289", "2127",
    "6305", "2768", "8697", "4385", "6532", "7832", "3088", "2587", "6586", "3769",
    # 中型 — 流動性高め
    "2929", "3665", "4385", "6532", "3697", "7342", "4477", "6200", "3923", "4485",
    "6035", "7816", "2158", "3064", "4751", "6526", "3436", "6146", "7741", "9449",
    "2181", "4716", "6448", "7272", "3116", "4248", "6966", "8698", "9143", "3141",
    "4776", "6460", "7729", "8136", "9602", "3626", "4565", "6588", "7747", "8154",
]

TARGET_DATES = [
    date(2015, 7, 19),
    date(2016, 3, 7),
    date(2017, 9, 2),
    date(2018, 12, 11),
    date(2019, 11, 23),
    date(2020, 8, 8),
    date(2021, 6, 15),
    date(2022, 4, 26),
    date(2023, 1, 30),
    date(2024, 2, 14),
]


def adjust_to_trading_day(d: date) -> date:
    """週末の場合、直前の金曜日に調整"""
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def download_all_prices(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """全銘柄の株価を一括ダウンロード"""
    tickers = [f"{c}.T" for c in codes]
    chunk_size = 200
    all_frames = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"  DL: chunk {i//chunk_size+1}/{(len(tickers)-1)//chunk_size+1} ({len(chunk)}銘柄)")
        try:
            data = yf.download(
                chunk, start=start, end=end,
                auto_adjust=True, threads=True, progress=False,
            )
            if data.empty:
                continue

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
                            all_frames.append(
                                df_single[["Date", "Open", "High", "Low", "Close", "Volume", "Code"]]
                                .dropna(subset=["Close"])
                            )
                    except (KeyError, TypeError):
                        continue
            else:
                data = data.reset_index()
                if len(chunk) == 1:
                    data["Code"] = chunk[0].replace(".T", "")
                    if "Close" in data.columns:
                        all_frames.append(
                            data[["Date", "Open", "High", "Low", "Close", "Volume", "Code"]]
                            .dropna(subset=["Close"])
                        )
        except Exception as e:
            print(f"    chunk error: {e}")
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


def run_for_date(
    cutoff: date, all_prices: pd.DataFrame, stocks: pd.DataFrame
) -> list:
    """指定日付でスクリーニング+スコアリング+リスク計算、Top3を返す"""
    cutoff_dt = pd.Timestamp(cutoff)
    start_dt = cutoff_dt - pd.Timedelta(days=400)

    prices_before = all_prices[
        (all_prices["Date"] >= start_dt) & (all_prices["Date"] <= cutoff_dt)
    ].copy()

    if prices_before.empty or prices_before["Code"].nunique() < 50:
        print(f"  データ不足: {prices_before['Code'].nunique() if not prices_before.empty else 0}銘柄")
        return []

    market_data = MarketData(
        stocks=stocks,
        prices=prices_before,
        financials=pd.DataFrame(),
        scan_date=cutoff,
    )

    try:
        pipeline = ScreeningPipeline()
        candidates = pipeline.run(market_data)
    except Exception as e:
        print(f"  スクリーニングエラー: {e}")
        return []

    try:
        scorer = MultiFactorScorer()
        scored_buy = scorer.score(candidates.buy, market_data, direction="buy")
    except Exception as e:
        print(f"  スコアリングエラー: {e}")
        return []

    # リスク計算
    risk_calc = RiskCalculator()
    for c in scored_buy:
        try:
            risk_calc.calculate(c, market_data)
        except Exception:
            pass

    top3 = sorted(scored_buy, key=lambda x: x.total_score, reverse=True)[:3]
    return top3


def main():
    t0 = time.time()

    print("=" * 70)
    print("指定10日付 おすすめ銘柄Top3 スクリーニング")
    print("=" * 70)

    # 日付を営業日に調整
    adjusted = []
    for d in TARGET_DATES:
        adj = adjust_to_trading_day(d)
        adjusted.append(adj)
        if adj != d:
            print(f"  {d} → {adj} (営業日に調整)")
        else:
            print(f"  {d}")

    # 銘柄一覧（ハードコード）
    print("\n[1/3] 銘柄一覧準備...")
    codes = list(dict.fromkeys(MAJOR_CODES))  # 重複除去
    # ダミーstocks DataFrame（スクリーニングで使う最低限のカラム）
    stocks = pd.DataFrame({
        "Code": codes,
        "CompanyName": [""] * len(codes),
        "MarketCodeName": ["プライム（内国株式）"] * len(codes),
        "Sector33CodeName": [""] * len(codes),
        "Sector17CodeName": [""] * len(codes),
    })
    print(f"  対象: {len(codes)}銘柄")

    # 全期間の株価を一括ダウンロード
    earliest = min(adjusted) - timedelta(days=500)
    latest = max(adjusted) + timedelta(days=1)
    print(f"\n[2/3] 株価データ一括ダウンロード ({earliest} ~ {latest})...")
    all_prices = download_all_prices(
        codes,
        start=earliest.isoformat(),
        end=latest.isoformat(),
    )
    if all_prices.empty:
        print("ERROR: 株価データ取得失敗")
        return 1
    print(f"  DL完了: {time.time()-t0:.0f}秒")

    # 各日付でスクリーニング
    print(f"\n[3/3] スクリーニング実行...")

    all_results = {}
    for cutoff in adjusted:
        print(f"\n{'='*70}")
        print(f"■ {cutoff} ({cutoff.strftime('%A')})")
        print(f"{'='*70}")

        top3 = run_for_date(cutoff, all_prices, stocks)
        all_results[cutoff] = top3

        if not top3:
            print("  候補なし")
            continue

        for rank, c in enumerate(top3, 1):
            print(f"\n  --- #{rank} {c.name} ({c.code}) ---")
            print(f"  総合スコア: {c.total_score:.1f}/100")
            print(f"  終値: ¥{c.close:,.0f}")
            print(f"  方向: {'買い' if c.direction == 'buy' else '売り'}")
            print(f"  セクター: {c.sector_name or '不明'}")
            print()
            print(f"  【スコア内訳】")
            print(f"    トレンド: {c.trend_score:.1f}  MACD: {c.macd_score:.1f}  出来高: {c.volume_score:.1f}")
            print(f"    ファンダ: {c.fundamental_score:.1f}  RSI: {c.rsi_score:.1f}  一目均衡表: {c.ichimoku_score:.1f}")
            print(f"    パターン: {c.pattern_score:.1f}  R/R: {c.risk_reward_score:.1f}  セクター: {c.sector_score:.1f}")
            print(f"    ニュース: {c.news_score:.1f}  需給: {c.margin_score:.1f}")
            print()
            print(f"  【保有推奨】")
            print(f"    推奨保有日数: {c.recommended_hold_days}日")
            print(f"    シグナル: {', '.join(c.signals[:5]) if c.signals else 'なし'}")
            if c.per:
                print(f"    PER: {c.per:.1f}  PBR: {c.pbr:.2f}  ROE: {c.roe:.1f}%" if c.pbr and c.roe else f"    PER: {c.per:.1f}")
            print()
            print(f"  【出口戦略】")
            print(f"    戦略: {c.exit_strategy or 'N/A'}")
            print(f"    損切り(SL): ¥{c.stop_loss:,.0f}" if c.stop_loss else "    損切り: N/A")
            print(f"    利確(TP): ¥{c.take_profit:,.0f}" if c.take_profit else "    利確: N/A")
            print(f"    部分利確: ¥{c.partial_exit_price:,.0f}" if c.partial_exit_price else "")
            print(f"    トレーリングストップ: ¥{c.trailing_stop_price:,.0f}" if c.trailing_stop_price else "")
            if c.risk_reward_ratio:
                print(f"    R/R比率: 1:{c.risk_reward_ratio:.1f}")
            if c.position_size_shares:
                print(f"    推奨ポジション: {c.position_size_shares}株 (¥{c.position_size_yen:,.0f})")
            if c.ruin_probability:
                print(f"    破産確率: {c.ruin_probability:.2f}%")

    elapsed = time.time() - t0
    print(f"\n\n総実行時間: {elapsed:.0f}秒")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
