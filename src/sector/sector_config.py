"""TOPIX-17セクター分類・景気サイクル・感応度定義

東証33業種をTOPIX-17セクターに対応付け、景気循環（セクターローテーション）、
為替・金利感応度、社会構造変化への影響度を定義する。

参考:
- JPX TOPIX-17シリーズ factsheet
- フィデリティ投信「景気サイクルとは」
- ブルーモ証券「セクターローテーションとは」
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ============================================================
# 景気サイクル4局面
# ============================================================
class MarketPhase(str, Enum):
    """景気サイクルの4局面（相場ステージ）"""
    KINYU = "金融相場"           # 不況後半→回復: 金融緩和で先行上昇
    GYOSEKI = "業績相場"         # 好況期: 企業業績が牽引
    GYAKU_KINYU = "逆金融相場"   # 過熱→後退初期: 利上げで逆風
    GYAKU_GYOSEKI = "逆業績相場"  # 後退→不況: 業績悪化


# ============================================================
# セクター特性分類
# ============================================================
class SectorType(str, Enum):
    """セクターの景気感応度分類"""
    CYCLICAL = "景気敏感"       # β > 1.0
    DEFENSIVE = "ディフェンシブ"  # β < 1.0
    GROWTH = "成長"             # DX・テクノロジー
    FINANCIAL = "金融"          # 金利敏感


# ============================================================
# TOPIX-17セクター定義
# ============================================================
@dataclass(frozen=True)
class SectorProfile:
    """セクターのプロファイル"""
    topix17_name: str           # TOPIX-17名称
    sector_type: SectorType     # 分類特性
    interest_rate_sensitivity: float  # 金利感応度 (-1.0〜+1.0, +は金利上昇で好影響)
    yen_weak_sensitivity: float      # 円安感応度 (-1.0〜+1.0, +は円安で好影響)
    beta: float                      # 市場ベータ（概算）
    favorable_phases: tuple[MarketPhase, ...]  # 有利な景気局面


# 東証33業種コード名 → TOPIX-17セクターのマッピング
SECTOR33_TO_TOPIX17: dict[str, str] = {
    # エネルギー資源
    "鉱業": "エネルギー資源",
    "石油・石炭製品": "エネルギー資源",
    # 素材・化学
    "化学": "素材・化学",
    "繊維製品": "素材・化学",
    "ガラス・土石製品": "素材・化学",
    # 鉄鋼・非鉄
    "鉄鋼": "鉄鋼・非鉄",
    "非鉄金属": "鉄鋼・非鉄",
    # 機械
    "機械": "機械",
    # 電気・精密
    "電気機器": "電気・精密",
    "精密機器": "電気・精密",
    # 輸送用機器
    "輸送用機器": "輸送用機器",
    # 食品
    "食料品": "食品",
    # 医薬品
    "医薬品": "医薬品",
    # 生活用品
    "パルプ・紙": "生活用品",
    "ゴム製品": "生活用品",
    "その他製品": "生活用品",
    # 小売
    "小売業": "小売",
    # 銀行
    "銀行業": "銀行",
    # 金融（除く銀行）
    "証券、商品先物取引業": "金融（除く銀行）",
    "保険業": "金融（除く銀行）",
    "その他金融業": "金融（除く銀行）",
    # 不動産
    "不動産業": "不動産",
    # 運輸・物流
    "陸運業": "運輸・物流",
    "海運業": "運輸・物流",
    "空運業": "運輸・物流",
    "倉庫・運輸関連業": "運輸・物流",
    # 情報通信・サービス他
    "情報・通信業": "情報通信・サービス他",
    "サービス業": "情報通信・サービス他",
    # 電力・ガス
    "電気・ガス業": "電力・ガス",
    # 建設・資材
    "建設業": "建設・資材",
    # その他（卸売は商社を含む特殊セクター）
    "卸売業": "卸売",
    "水産・農林業": "食品",
    "金属製品": "鉄鋼・非鉄",
}


# TOPIX-17セクタープロファイル
TOPIX17_PROFILES: dict[str, SectorProfile] = {
    "エネルギー資源": SectorProfile(
        topix17_name="エネルギー資源",
        sector_type=SectorType.CYCLICAL,
        interest_rate_sensitivity=0.2,
        yen_weak_sensitivity=0.5,
        beta=1.2,
        favorable_phases=(MarketPhase.GYAKU_KINYU,),
    ),
    "素材・化学": SectorProfile(
        topix17_name="素材・化学",
        sector_type=SectorType.CYCLICAL,
        interest_rate_sensitivity=0.0,
        yen_weak_sensitivity=0.3,
        beta=1.1,
        favorable_phases=(MarketPhase.GYOSEKI,),
    ),
    "鉄鋼・非鉄": SectorProfile(
        topix17_name="鉄鋼・非鉄",
        sector_type=SectorType.CYCLICAL,
        interest_rate_sensitivity=0.0,
        yen_weak_sensitivity=0.2,
        beta=1.3,
        favorable_phases=(MarketPhase.GYOSEKI,),
    ),
    "機械": SectorProfile(
        topix17_name="機械",
        sector_type=SectorType.CYCLICAL,
        interest_rate_sensitivity=-0.1,
        yen_weak_sensitivity=0.6,
        beta=1.2,
        favorable_phases=(MarketPhase.GYOSEKI,),
    ),
    "電気・精密": SectorProfile(
        topix17_name="電気・精密",
        sector_type=SectorType.GROWTH,
        interest_rate_sensitivity=-0.4,
        yen_weak_sensitivity=0.5,
        beta=1.2,
        favorable_phases=(MarketPhase.KINYU, MarketPhase.GYOSEKI),
    ),
    "輸送用機器": SectorProfile(
        topix17_name="輸送用機器",
        sector_type=SectorType.CYCLICAL,
        interest_rate_sensitivity=-0.1,
        yen_weak_sensitivity=0.8,
        beta=1.1,
        favorable_phases=(MarketPhase.GYOSEKI,),
    ),
    "食品": SectorProfile(
        topix17_name="食品",
        sector_type=SectorType.DEFENSIVE,
        interest_rate_sensitivity=0.0,
        yen_weak_sensitivity=-0.4,
        beta=0.7,
        favorable_phases=(MarketPhase.GYAKU_GYOSEKI,),
    ),
    "医薬品": SectorProfile(
        topix17_name="医薬品",
        sector_type=SectorType.DEFENSIVE,
        interest_rate_sensitivity=-0.1,
        yen_weak_sensitivity=-0.1,
        beta=0.6,
        favorable_phases=(MarketPhase.GYAKU_GYOSEKI,),
    ),
    "生活用品": SectorProfile(
        topix17_name="生活用品",
        sector_type=SectorType.DEFENSIVE,
        interest_rate_sensitivity=0.0,
        yen_weak_sensitivity=-0.3,
        beta=0.8,
        favorable_phases=(MarketPhase.GYAKU_GYOSEKI,),
    ),
    "小売": SectorProfile(
        topix17_name="小売",
        sector_type=SectorType.DEFENSIVE,
        interest_rate_sensitivity=-0.1,
        yen_weak_sensitivity=-0.2,
        beta=0.9,
        favorable_phases=(MarketPhase.GYAKU_GYOSEKI, MarketPhase.KINYU),
    ),
    "銀行": SectorProfile(
        topix17_name="銀行",
        sector_type=SectorType.FINANCIAL,
        interest_rate_sensitivity=0.9,
        yen_weak_sensitivity=0.1,
        beta=1.1,
        favorable_phases=(MarketPhase.GYAKU_KINYU, MarketPhase.GYOSEKI),
    ),
    "金融（除く銀行）": SectorProfile(
        topix17_name="金融（除く銀行）",
        sector_type=SectorType.FINANCIAL,
        interest_rate_sensitivity=0.5,
        yen_weak_sensitivity=0.1,
        beta=1.2,
        favorable_phases=(MarketPhase.KINYU, MarketPhase.GYOSEKI),
    ),
    "不動産": SectorProfile(
        topix17_name="不動産",
        sector_type=SectorType.FINANCIAL,
        interest_rate_sensitivity=-0.7,
        yen_weak_sensitivity=0.0,
        beta=1.1,
        favorable_phases=(MarketPhase.KINYU,),
    ),
    "運輸・物流": SectorProfile(
        topix17_name="運輸・物流",
        sector_type=SectorType.CYCLICAL,
        interest_rate_sensitivity=0.0,
        yen_weak_sensitivity=-0.2,
        beta=1.0,
        favorable_phases=(MarketPhase.GYOSEKI,),
    ),
    "情報通信・サービス他": SectorProfile(
        topix17_name="情報通信・サービス他",
        sector_type=SectorType.GROWTH,
        interest_rate_sensitivity=-0.5,
        yen_weak_sensitivity=0.0,
        beta=1.0,
        favorable_phases=(MarketPhase.KINYU,),
    ),
    "電力・ガス": SectorProfile(
        topix17_name="電力・ガス",
        sector_type=SectorType.DEFENSIVE,
        interest_rate_sensitivity=0.1,
        yen_weak_sensitivity=-0.5,
        beta=0.5,
        favorable_phases=(MarketPhase.GYAKU_GYOSEKI,),
    ),
    "建設・資材": SectorProfile(
        topix17_name="建設・資材",
        sector_type=SectorType.CYCLICAL,
        interest_rate_sensitivity=-0.3,
        yen_weak_sensitivity=0.0,
        beta=0.9,
        favorable_phases=(MarketPhase.KINYU, MarketPhase.GYOSEKI),
    ),
    "卸売": SectorProfile(
        topix17_name="卸売",
        sector_type=SectorType.CYCLICAL,
        interest_rate_sensitivity=0.1,
        yen_weak_sensitivity=0.4,
        beta=1.0,
        favorable_phases=(MarketPhase.GYOSEKI, MarketPhase.GYAKU_KINYU),
    ),
}


def get_topix17_sector(sector33_name: str) -> str | None:
    """東証33業種名からTOPIX-17セクター名を取得"""
    return SECTOR33_TO_TOPIX17.get(sector33_name)


def get_sector_profile(topix17_name: str) -> SectorProfile | None:
    """TOPIX-17セクター名からプロファイルを取得"""
    return TOPIX17_PROFILES.get(topix17_name)
