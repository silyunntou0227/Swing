"""ファンダメンタルスクリーニングモジュール（Layer 1）"""

from __future__ import annotations

import pandas as pd

from src.config import (
    PER_MIN, PER_MAX, PBR_MIN, PBR_MAX,
    ROE_MIN, EPS_GROWTH_MIN, MARGIN_RATIO_MAX,
)
from src.utils.logging_config import logger


def filter_fundamentals(
    stocks: pd.DataFrame,
    financials: pd.DataFrame,
) -> pd.DataFrame:
    """ファンダメンタルフィルタを適用

    Args:
        stocks: 銘柄一覧（Code列を含む）
        financials: 財務データ

    Returns:
        フィルタ通過した銘柄のDataFrame
    """
    if financials.empty:
        logger.warning("財務データなし — ファンダメンタルフィルタをスキップ")
        return stocks

    # 銘柄一覧と財務データを結合
    merged = stocks.copy()
    if "Code" in financials.columns:
        # 最新の財務データとマージ
        fin_cols = _select_financial_columns(financials)
        merged = merged.merge(fin_cols, on="Code", how="left")

    initial_count = len(merged)

    # PERフィルタ
    if "per" in merged.columns:
        merged = merged[
            (merged["per"].isna())  # 財務データなしは通過
            | ((merged["per"] >= PER_MIN) & (merged["per"] <= PER_MAX))
        ]

    # PBRフィルタ
    if "pbr" in merged.columns:
        merged = merged[
            (merged["pbr"].isna())
            | ((merged["pbr"] >= PBR_MIN) & (merged["pbr"] <= PBR_MAX))
        ]

    # ROEフィルタ
    if "roe" in merged.columns:
        merged = merged[
            (merged["roe"].isna())
            | (merged["roe"] >= ROE_MIN)
        ]

    # EPS成長率フィルタ
    if "eps_growth" in merged.columns:
        merged = merged[
            (merged["eps_growth"].isna())
            | (merged["eps_growth"] >= EPS_GROWTH_MIN)
        ]

    logger.info(
        f"ファンダメンタルフィルタ: {initial_count} → {len(merged)}銘柄"
    )
    return merged


def _select_financial_columns(financials: pd.DataFrame) -> pd.DataFrame:
    """財務DataFrameから必要な列を抽出・計算"""
    df = financials.copy()
    result = pd.DataFrame()
    result["Code"] = df["Code"]

    # PER計算
    if "EarningsPerShare" in df.columns:
        eps = pd.to_numeric(df["EarningsPerShare"], errors="coerce")
        # PERはClose/EPSで計算するが、ここではJ-Quantsの値を使う
        result["eps"] = eps

    # PBR計算
    if "BookValuePerShare" in df.columns:
        result["bvps"] = pd.to_numeric(df["BookValuePerShare"], errors="coerce")

    # ROE
    if "ReturnOnEquity" in df.columns:
        result["roe"] = pd.to_numeric(df["ReturnOnEquity"], errors="coerce")
    elif "Profit" in df.columns and "NetAssets" in df.columns:
        profit = pd.to_numeric(df["Profit"], errors="coerce")
        net_assets = pd.to_numeric(df["NetAssets"], errors="coerce")
        result["roe"] = (profit / net_assets * 100).where(net_assets > 0)

    # EPS成長率（前年比）
    # J-Quantsの財務データに含まれる場合
    if "ForecastEarningsPerShare" in df.columns and "EarningsPerShare" in df.columns:
        forecast_eps = pd.to_numeric(df["ForecastEarningsPerShare"], errors="coerce")
        actual_eps = pd.to_numeric(df["EarningsPerShare"], errors="coerce")
        result["eps_growth"] = ((forecast_eps - actual_eps) / actual_eps.abs() * 100).where(
            actual_eps.abs() > 0
        )

    return result


def calculate_fundamental_score(
    code: str,
    financials: pd.DataFrame,
    close: float,
) -> tuple[float, dict]:
    """個別銘柄のファンダメンタルスコアを計算

    Args:
        code: 銘柄コード
        financials: 財務データ
        close: 現在の終値

    Returns:
        (score 0-100, details dict)
    """
    if financials.empty:
        return 50.0, {}

    fin = financials[financials["Code"] == code]
    if fin.empty:
        return 50.0, {}

    row = fin.iloc[0]
    score = 50.0  # ベース
    details = {}

    # PER評価
    eps = pd.to_numeric(row.get("EarningsPerShare", None), errors="coerce")
    if pd.notna(eps) and eps > 0 and close > 0:
        per = close / eps
        details["per"] = round(per, 1)
        if 8 <= per <= 15:
            score += 15  # 割安
        elif 15 < per <= 25:
            score += 5   # 適正
        elif per > 35:
            score -= 10  # 割高

    # PBR評価
    bvps = pd.to_numeric(row.get("BookValuePerShare", None), errors="coerce")
    if pd.notna(bvps) and bvps > 0 and close > 0:
        pbr = close / bvps
        details["pbr"] = round(pbr, 2)
        if 0.5 <= pbr <= 1.0:
            score += 15  # 割安
        elif 1.0 < pbr <= 2.0:
            score += 5
        elif pbr > 3.0:
            score -= 5

    # ROE評価
    roe = pd.to_numeric(row.get("ReturnOnEquity", None), errors="coerce")
    if pd.notna(roe):
        details["roe"] = round(roe, 1)
        if roe >= 15:
            score += 15
        elif roe >= 10:
            score += 10
        elif roe >= 5:
            score += 5

    return max(0, min(100, score)), details
