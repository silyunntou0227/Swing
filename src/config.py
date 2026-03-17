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
# V1互換用（非推奨）
JQUANTS_MAIL = os.environ.get("JQUANTS_MAIL", "")
JQUANTS_PASSWORD = os.environ.get("JQUANTS_PASSWORD", "")
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
ADX_MIN = 20.0
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
EMA_SHORT = 12
EMA_LONG = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
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
# TDnet適時開示スコア
DISCLOSURE_SCORES = {
    "業績上方修正": 12,
    "増配": 8,
    "自社株買い": 8,
    "株式分割": 5,
    "業績下方修正": -18,
    "減配": -10,
    "MBO": -999,  # 除外
    "TOB": -999,  # 除外
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
@dataclass
class ScoringWeights:
    trend: float = 0.18
    macd: float = 0.12
    volume: float = 0.12
    fundamental: float = 0.13
    rsi: float = 0.08
    ichimoku: float = 0.08
    pattern: float = 0.08
    risk_reward: float = 0.05
    news_disclosure: float = 0.10
    margin_supply: float = 0.06

    def validate(self) -> bool:
        total = sum([
            self.trend, self.macd, self.volume, self.fundamental,
            self.rsi, self.ichimoku, self.pattern, self.risk_reward,
            self.news_disclosure, self.margin_supply,
        ])
        return abs(total - 1.0) < 0.001


SCORING_WEIGHTS = ScoringWeights()

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
