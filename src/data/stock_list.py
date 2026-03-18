"""東証上場銘柄一覧取得モジュール

J-Quants 無料プランでは /equities/listed が 403 のため、
JPX（日本取引所グループ）公式サイトの銘柄一覧CSVを使用する。

ソース: https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
直リンク: https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from src.utils.logging_config import logger

# JPX公式の銘柄一覧（Excel/CSV形式）
JPX_STOCK_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)

# カラム名マッピング（日本語 → 内部名）
COLUMN_MAP = {
    "コード": "Code",
    "銘柄名": "CompanyName",
    "市場・商品区分": "MarketCodeName",
    "33業種コード": "Sector33Code",
    "33業種区分": "Sector33CodeName",
    "17業種コード": "Sector17Code",
    "17業種区分": "Sector17CodeName",
    "規模コード": "ScaleCode",
    "規模区分": "ScaleCategory",
}

# 除外対象キーワード
EXCLUDE_KEYWORDS = ["ETF", "REIT", "ETN", "インフラファンド", "出資証券"]

# 対象市場
TARGET_MARKETS = ["プライム", "スタンダード", "グロース"]


def fetch_jpx_stock_list() -> pd.DataFrame:
    """JPX公式サイトから全上場銘柄一覧を取得

    Returns:
        columns: Code, CompanyName, MarketCodeName,
                 Sector33CodeName, Sector17CodeName, etc.
    """
    logger.info("JPX公式: 上場銘柄一覧を取得中...")

    try:
        resp = requests.get(JPX_STOCK_LIST_URL, timeout=30)
        resp.raise_for_status()

        # JPXのファイルはExcel形式（.xls）
        df = pd.read_excel(
            io.BytesIO(resp.content),
            dtype={"コード": str},
        )

        # カラム名を変換
        df = df.rename(columns=COLUMN_MAP)

        # 必須カラムの検証
        required = {"Code", "CompanyName", "MarketCodeName"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(
                f"JPX銘柄一覧: 必須カラム不足 {missing} — "
                f"JPXのファイル形式が変更された可能性があります。"
                f"現在のカラム: {list(df.columns[:10])}"
            )
            # カラム名が変更されていても、位置ベースのフォールバックを試みる
            if len(df.columns) >= 3 and "Code" not in df.columns:
                logger.info("JPX銘柄一覧: 位置ベースのカラム推定を試行")
                cols = list(df.columns)
                fallback_map = {}
                for col in cols:
                    sample = str(df[col].iloc[0]) if len(df) > 0 else ""
                    if sample.isdigit() and len(sample) in (4, 5):
                        fallback_map[col] = "Code"
                    elif "プライム" in str(df[col].values[:5]) or "スタンダード" in str(df[col].values[:5]):
                        fallback_map[col] = "MarketCodeName"
                if fallback_map:
                    df = df.rename(columns=fallback_map)
                    logger.info(f"JPX銘柄一覧: フォールバックマッピング適用: {fallback_map}")

        # 銘柄コードを4桁に正規化
        if "Code" in df.columns:
            df["Code"] = df["Code"].astype(str).str[:4]

        logger.info(f"JPX公式: {len(df)}銘柄取得")
        return df

    except requests.exceptions.Timeout:
        logger.warning("JPX銘柄一覧: タイムアウト（30秒）")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"JPX銘柄一覧取得失敗: {type(e).__name__}: {e}")
        return pd.DataFrame()


def get_tradeable_codes(df: pd.DataFrame, max_stocks: int = 0) -> list[str]:
    """売買対象となる銘柄コードのリストを返す

    ETF/REIT/ETN等を除外し、プライム/スタンダード/グロース市場の普通株のみ。

    Args:
        df: 銘柄一覧DataFrame
        max_stocks: 最大銘柄数（0=制限なし）。大規模市場から優先的に選択。
    """
    if df.empty or "Code" not in df.columns:
        return []

    filtered = df.copy()

    # 市場区分フィルタ（プライム・スタンダード・グロースのみ）
    if "MarketCodeName" in filtered.columns:
        mask = filtered["MarketCodeName"].apply(
            lambda x: any(m in str(x) for m in TARGET_MARKETS)
        )
        filtered = filtered[mask]

    # ETF/REIT等を除外
    if "MarketCodeName" in filtered.columns:
        for kw in EXCLUDE_KEYWORDS:
            filtered = filtered[
                ~filtered["MarketCodeName"].str.contains(kw, na=False)
            ]
    if "Sector17CodeName" in filtered.columns:
        for kw in EXCLUDE_KEYWORDS:
            filtered = filtered[
                ~filtered["Sector17CodeName"].str.contains(kw, na=False)
            ]

    # max_stocks が指定されている場合、プライム→スタンダード→グロースの優先順で絞る
    if max_stocks > 0 and "MarketCodeName" in filtered.columns:
        priority_order = ["プライム", "スタンダード", "グロース"]
        selected = []
        for market in priority_order:
            market_stocks = filtered[
                filtered["MarketCodeName"].str.contains(market, na=False)
            ]
            remaining = max_stocks - len(selected)
            if remaining <= 0:
                break
            selected.append(market_stocks.head(remaining))

        if selected:
            filtered = pd.concat(selected, ignore_index=True)

    codes = filtered["Code"].unique().tolist()
    logger.info(f"売買対象銘柄: {len(codes)}銘柄")
    return codes
