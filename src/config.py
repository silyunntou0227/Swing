"""全設定パラメータ集約モジュール"""

import os
from dataclasses import dataclass, field

# ローカル実行時に .env ファイルから環境変数を自動読み込み
# GitHub Actions では .env が存在しないため無視される
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv が無い環境（GitHub Actions）ではスキップ


# ============================================================
# API認証情報（環境変数から取得）
# ============================================================
JQUANTS_API_KEY = os.environ.get("JQUANTS_API_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
LINE_CHANNEL_TOKEN = os.environ.get("LINE_CHANNEL_TOKEN", "")
LINE_USER_ID = os.environ.get("LINE_USER_ID", "")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
EDINET_API_KEY = os.environ.get("EDINET_API_KEY", "")

# ============================================================
# データ取得設定
# ============================================================
HISTORY_DAYS = 200  # 取得する営業日数（約10ヶ月分）
JQUANTS_RATE_LIMIT = 5  # req/min（無料プラン）

# ============================================================
# ユニバースフィルタ（Layer 0）
# ============================================================
TARGET_MARKETS = ["プライム", "スタンダード", "グロース"]
EXCLUDE_CATEGORIES = ["ETF", "REIT", "ETN", "インフラファンド"]
MIN_LISTING_DAYS = 30  # 新規上場除外日数

# ============================================================
# ファンダメンタルフィルタ（Layer 1）
# ============================================================
PER_MIN = 5.0
PER_MAX = 40.0
PBR_MIN = 0.3
PBR_MAX = 5.0
ROE_MIN = 5.0  # %
EPS_GROWTH_MIN = -10.0  # % (YoY)
MARGIN_RATIO_MAX = 5.0  # 信用倍率上限

# ============================================================
# トレンド・流動性フィルタ（Layer 2）
# ============================================================
ADX_MIN = 25.0  # 20→25: 研究に基づきトレンド確認精度向上
VOLUME_AVG_MIN = 50000  # 20日平均出来高（株）
TURNOVER_MIN = 50_000_000  # 売買代金（円/日）

# ============================================================
# エントリーシグナル（Layer 3）
# ============================================================
SIGNAL_LOOKBACK_DAYS = 3  # シグナル検出の遡り日数
VOLUME_SPIKE_RATIO = 1.5  # 出来高急増の倍率（vs 20日平均）

# ============================================================
# テクニカル指標パラメータ
# ============================================================
SMA_SHORT = 5
SMA_MEDIUM = 25
SMA_LONG = 75
SMA_VERY_LONG = 200
# MACD: Kang(2021)論文に基づき日経225最適パラメータ (4,22,3) を採用
EMA_SHORT = 4    # 12→4: 日本市場の速い周期に対応
EMA_LONG = 22    # 26→22: 日経225最適化
MACD_SIGNAL = 3  # 9→3: 偽シグナル削減

# RSI: Connors RSI研究に基づき短期RSI(2)を追加、従来RSI(14)も併用
RSI_PERIOD = 14          # 従来RSI（中長期トレンド確認用）
RSI_PERIOD_SHORT = 2     # Connors RSI（短期エントリー用）
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_SHORT_OVERSOLD = 10  # RSI(2)用: 極端な売られすぎ
RSI_SHORT_OVERBOUGHT = 90  # RSI(2)用: 極端な買われすぎ
STOCH_K = 14
STOCH_D = 3
STOCH_SMOOTH = 3
ATR_PERIOD = 14

# 一目均衡表
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKOU_B = 52

# ============================================================
# ニュース・開示スコアリング（Layer 4）
# ============================================================
# 除外スコア（MBO/TOBなど、スクリーニングから強制除外する銘柄用）
EXCLUSION_SCORE = -999

# TDnet適時開示スコア
DISCLOSURE_SCORES = {
    "業績上方修正": 12,
    "増配": 8,
    "自社株買い": 8,
    "株式分割": 5,
    "業績下方修正": -18,
    "減配": -10,
    "MBO": EXCLUSION_SCORE,
    "TOB": EXCLUSION_SCORE,
}

# EDINET開示スコア
EDINET_SCORES = {
    "大量保有報告書_新規": 5,
    "大量保有報告書_増加": 3,
    "大量保有報告書_減少": -3,
}

# ニュースセンチメント辞書
POSITIVE_KEYWORDS = [
    "増益", "最高益", "上方修正", "好調", "受注増", "新製品",
    "黒字転換", "増配", "自社株買い", "過去最高", "シェア拡大",
    "業績好調", "売上増", "利益増", "成長加速", "需要増",
    "新規受注", "契約獲得", "提携", "M&A", "TOPIXウェイト増",
]

NEGATIVE_KEYWORDS = [
    "減益", "下方修正", "不正", "リコール", "訴訟", "債務超過",
    "赤字", "減配", "無配", "倒産", "民事再生", "業績悪化",
    "売上減", "在庫増", "引当金", "特別損失", "行政処分",
    "上場廃止", "監理銘柄", "粉飾", "横領", "データ改ざん",
]

NEWS_SCORE_MAX = 10
NEWS_SCORE_MIN = -10

# マクロ環境スコア
MACRO_SCORE_MAX = 5
MACRO_SCORE_MIN = -5

# ============================================================
# スコアリングウェイト
# ============================================================
# 保有期間予測パラメータ
MAX_HOLDING_DAYS = 10         # 最大保有日数
TRAILING_STOP_ATR_MULT = 2.5  # トレーリングストップ: ATR×2.5
PROFIT_TARGET_ATR_MULT = 3.0  # 利確目標: ATR×3.0
PARTIAL_EXIT_ATR_MULT = 1.5   # 部分利確: ATR×1.5（50%決済）

# ============================================================
# スコアリングウェイト（研究ベース最適化）
# ============================================================
# 多因子モデル研究に基づきトレンド・モメンタム重視に再配分
# トレンド系(trend+ichimoku): 35%, モメンタム系(macd+rsi): 25%
# ボリューム系: 15%, ファンダ: 10%, その他: 15%
@dataclass
class ScoringWeights:
    trend: float = 0.20       # 22→20: セクター追加に伴い再配分
    macd: float = 0.14        # 15→14
    volume: float = 0.11      # 12→11
    fundamental: float = 0.10  # 据え置き
    rsi: float = 0.10         # 据え置き
    ichimoku: float = 0.12    # 13→12
    pattern: float = 0.05     # 据え置き
    risk_reward: float = 0.05  # 据え置き
    news_disclosure: float = 0.05  # 据え置き
    margin_supply: float = 0.03    # 据え置き
    sector: float = 0.05      # NEW: セクターローテーション分析

    def validate(self) -> bool:
        total = sum([
            self.trend, self.macd, self.volume, self.fundamental,
            self.rsi, self.ichimoku, self.pattern, self.risk_reward,
            self.news_disclosure, self.margin_supply, self.sector,
        ])
        return abs(total - 1.0) < 0.001


SCORING_WEIGHTS = ScoringWeights()
assert SCORING_WEIGHTS.validate(), "スコアリングウェイトの合計が1.0ではありません"

# ============================================================
# リスク管理
# ============================================================
RISK_PER_TRADE = 0.01  # 1トレードあたりのリスク（総資金比）
DEFAULT_CAPITAL = 1_000_000  # デフォルト運用資金（円）
STOP_LOSS_ATR_MULTIPLIER = 2.0  # ATR × N で損切りライン
TAKE_PROFIT_RR_RATIO = 2.0  # 最低R:R比率
BALSARA_WIN_RATE = 0.45  # デフォルト勝率（保守的）
BALSARA_PAYOFF_RATIO = 2.0  # デフォルト損益比

# ============================================================
# 出力設定
# ============================================================
TOP_BUY_CANDIDATES = 10
TOP_SELL_CANDIDATES = 5

# ============================================================
# フィボナッチ水準
# ============================================================
FIBONACCI_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]

# ============================================================
# ログ設定
# ============================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
