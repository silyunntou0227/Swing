# Swing - 日本株スイングトレード自動スクリーニングシステム

東証上場 約4000銘柄を毎日自動スキャンし、スイングトレード（2日〜2週間）の
買い/売り候補をスコアリングして Discord に通知するシステム。

GitHub Actions で毎日自動実行。**ランニングコスト0円**。

---

## システム概要

```
JPX銘柄一覧 ─┐
yfinance     ─┤
J-Quants V2  ─┤──→ 5層スクリーニング ──→ スコアリング ──→ Discord通知
TDnet開示    ─┤      (4000→10銘柄)     (0-100点)      📈買い候補Top10
EDINET       ─┤                                        📉売り候補Top5
ニュースAPI  ─┤
日経平均     ─┘
```

### 5層スクリーニングパイプライン

| Layer | 名称 | 処理内容 | 通過目安 |
|-------|------|----------|----------|
| 0 | ユニバース | プライム/スタンダード/グロース、ETF・REIT除外 | 4000→1200 |
| 1 | ファンダメンタル | PER 5-40、PBR 0.3-5.0、ROE 5%+ | →1000 |
| 2+3 | テクニカル+シグナル | ADX≧20、流動性、ゴールデンクロス等 | →50-100 |
| 4 | ニュース・開示 | MBO/TOB除外、業績修正反映 | →30-50 |

### 10ファクタースコアリング (0-100点)

| ファクター | ウェイト | データソース |
|-----------|---------|-------------|
| トレンド（SMA配列） | 18% | yfinance |
| MACD | 12% | yfinance |
| 出来高 | 12% | yfinance |
| ファンダメンタル | 13% | J-Quants / yfinance |
| RSI | 8% | yfinance |
| 一目均衡表 | 8% | yfinance |
| ローソク足パターン | 8% | yfinance |
| リスク/リワード | 5% | yfinance |
| ニュース・開示 | 10% | TDnet / EDINET / NewsAPI |
| 需給（信用残） | 6% | Yahoo Finance |

---

## データソース

| データ | 主力ソース | フォールバック | 用途 |
|--------|-----------|---------------|------|
| 銘柄一覧 | JPX公式CSV | J-Quants V2 | ユニバース定義 |
| 株価OHLCV | **yfinance** | J-Quants V2 | テクニカル指標 |
| 財務データ | J-Quants V2 | スキップ | PER/PBR/ROE |
| 適時開示 | TDnet | — | 業績修正・自社株買い |
| 大量保有報告 | EDINET API | — | 機関投資家動向 |
| ニュース | NewsAPI / Google RSS | — | センチメント |
| マクロ指標 | yfinance (^N225) | — | 市場環境補正 |

> **J-Quants 無料プランの制限**（5 req/min, 一部403）を回避するため、
> yfinance を株価データの主力とし、J-Quants は補助的に使用。

---

## セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/silyunntou0227/Swing.git
cd Swing
```

### 2. 依存パッケージをインストール

```bash
pip install -r requirements.txt
pip install python-dotenv  # ローカル実行用
```

### 3. 環境変数を設定

`.env` ファイルを作成（`.gitignore` に含まれるため安全）：

```env
# 必須
JQUANTS_API_KEY=your_jquants_api_key
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# オプション（なくても動作）
NEWS_API_KEY=
EDINET_API_KEY=
LINE_CHANNEL_TOKEN=
LINE_USER_ID=
```

### 4. 実行

```bash
# ローカル実行
python -m src.main

# テスト実行
python -m pytest tests/ -v
```

---

## GitHub Actions（自動実行）

### 毎日自動スキャン

`daily_scan.yml` — 毎日 JST 18:00（東証引け後）に自動実行

### 手動スキャン

`manual_scan.yml` — GitHub Actions 画面から手動実行可能

### GitHub Secrets に設定が必要な環境変数

| Secret名 | 必須 | 説明 |
|-----------|------|------|
| `JQUANTS_API_KEY` | ✅ | J-Quants API V2 キー |
| `DISCORD_WEBHOOK_URL` | ✅ | Discord Webhook URL |
| `NEWS_API_KEY` | — | NewsAPI.org キー |
| `EDINET_API_KEY` | — | EDINET API キー |
| `LINE_CHANNEL_TOKEN` | — | LINE Messaging API |
| `LINE_USER_ID` | — | LINE 通知先ユーザーID |

---

## プロジェクト構成

```
src/
├── main.py                 # エントリーポイント（5ステップ実行）
├── config.py               # 全設定パラメータ集約
├── data/                   # データ取得層
│   ├── data_loader.py      # 全ソース統合ローダー
│   ├── stock_list.py       # JPX公式 銘柄一覧取得
│   ├── yahoo_client.py     # yfinance 株価一括取得（主力）
│   ├── jquants_client.py   # J-Quants API V2（補助）
│   ├── tdnet_client.py     # TDnet 適時開示
│   ├── edinet_client.py    # EDINET 大量保有報告書
│   ├── news_client.py      # ニュースセンチメント
│   ├── macro_client.py     # マクロ指標（日経平均）
│   ├── margin_client.py    # 信用残・空売り
│   └── market_calendar.py  # 日本市場カレンダー
├── screening/              # 5層スクリーニング
│   ├── pipeline.py         # パイプラインオーケストレーター
│   ├── fundamental.py      # Layer 1: ファンダメンタル
│   ├── liquidity.py        # Layer 2: 流動性
│   ├── news_filter.py      # Layer 4: ニュース・開示
│   └── __init__.py
├── indicators/             # テクニカル指標
│   ├── technical.py        # 指標集約モジュール
│   ├── trend.py            # SMA, EMA, MACD, ADX, 一目均衡表
│   ├── oscillator.py       # RSI, ストキャスティクス
│   ├── volume.py           # OBV, 出来高分析
│   ├── pattern.py          # ローソク足パターン
│   └── wave.py             # フィボナッチ, エリオット波動
├── scoring/                # スコアリング
│   ├── scorer.py           # 10ファクター加重スコアリング
│   └── risk.py             # リスク管理・ポジションサイジング
└── notify/                 # 通知
    ├── discord.py          # Discord Webhook
    ├── formatter.py        # 結果フォーマット（Embed）
    └── line.py             # LINE通知
```

---

## Discord 通知サンプル

```
📈 買い候補 #1: トヨタ自動車 (7203)
スコア: 78/100 | 現在値: ¥2,850

シグナル: ゴールデンクロス(5/25), MACD買い, RSI反転上昇
テクニカル: トレンド 90 | MACD 80 | RSI 65 | 一目 70
ファンダメンタル: PER 12.3 | PBR 1.05 | ROE 12.8%
リスク管理: 損切り ¥2,750 (-3.5%) | 利確 ¥3,050 (+7.0%)
           推奨 300株 (¥855,000) | R:R 1:2.0 | 破産確率 0.8%
```

---

## テクニカル指標一覧

### トレンド系
- 単純移動平均線 (SMA 5/25/75/200)
- 指数移動平均線 (EMA 12/26)
- MACD (12/26/9)
- ADX (14)
- 一目均衡表 (9/26/52)
- グランビルの法則

### オシレーター系
- RSI (14)
- ストキャスティクス (%K14, %D3)
- ダイバージェンス検出

### 出来高系
- OBV (On-Balance Volume)
- 出来高急増検出 (1.5倍ルール)
- 出来高比率

### パターン系
- 包み足（強気/弱気）
- はらみ足
- ハンマー / シューティングスター
- 三兵 / 三羽烏

### 波動分析
- フィボナッチ・リトレースメント
- ダウ理論トレンド判定

---

## リスク管理

- **損切り**: ATR × 2.0（エントリー価格から）
- **利確**: ATR × 4.0（R:R = 1:2.0）
- **ポジションサイズ**: 1トレードあたり資金の1%リスク
- **バルサラの破産確率**: 勝率45%、損益比2.0で計算

---

## 技術スタック

| 区分 | 技術 |
|------|------|
| 言語 | Python 3.11+ |
| テクニカル指標 | `ta` ライブラリ |
| 株価データ | `yfinance` |
| Web スクレイピング | `beautifulsoup4` + `lxml` |
| RSS パース | `feedparser` |
| CI/CD | GitHub Actions |
| 通知 | Discord Webhook / LINE Messaging API |

---

## ライセンス

Private repository — 個人利用のみ
