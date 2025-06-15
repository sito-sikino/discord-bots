# LangGraph Multi-Agent Discord Bot System v0.3

Google Gemini 2.0 FlashモデルをベースにしたLangGraph CLIチャットボット。
**将来的にDiscord VPS常時稼働ボットへの展開を想定**。

## システム概要

### 🤖 エージェント構成
- **Spectra**: コミュニケーション・説明・対話促進専門
- **LynQ**: 論理分析・構造化思考・概念明確化専門  
- **Paz**: 創造的発想・可能性探索・発散思考専門

### 🧠 メモリシステム
- **短期メモリ**: MemorySaver（セッション会話履歴）
- **長期メモリ**: PostgreSQL + pgvector（永続化 + セマンティック検索）
- **Obsidian統合**: sito-sikino/Obsidianリポジトリの全ノート自動読み込み

### 🎯 ルーティング精度
**95%達成済み** - Pure LLM-driven routing system

## 技術仕様

### 使用技術
- **LLM**: Gemini 2.0 Flash (gemini-2.0-flash)
- **フレームワーク**: LangGraph v0.4.8 + LangChain v0.3.25
- **言語**: Python 3.9+
- **データベース**: PostgreSQL + pgvector
- **埋め込み**: models/text-embedding-004 (768次元)

### 依存パッケージ
```
langchain==0.3.25
langgraph==0.4.8
langchain-google-genai==2.1.5
langgraph-checkpoint-postgres==2.0.21
psycopg==3.2.0
```

## セットアップ

### 1. 環境準備
```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存関係インストール
pip install -r requirements.txt
```

### 2. PostgreSQL起動
```bash
# Docker Composeでpostgres起動
docker compose up -d postgres
```

### 3. 環境変数設定
```bash
# .envファイル作成
export GOOGLE_API_KEY="your_gemini_api_key"
export GITHUB_TOKEN="your_github_token"  # Obsidian統合用
```

### 4. 実行
```bash
# CLIチャットボット起動
python main.py
```

## 使用方法

### 基本コマンド
- **通常会話**: 任意のテキスト入力
- **長期メモリ保存**: `!私の名前はシトです`（!で始める）
- **メモリ一覧**: `memories`
- **終了**: `Ctrl+C`

### GitHub連携（MCP統合）
- **ヘルプ**: `gh:help`
- **リポジトリ検索**: `gh:search langchain`
- **リポジトリ詳細**: `gh:microsoft/vscode`
- **ファイル閲覧**: `gh:browse microsoft/vscode src`
- **ファイル読み取り**: `gh:read microsoft/vscode/README.md`

## アーキテクチャ

### ディレクトリ構造
```
project-005/
├── CLAUDE.md           # 開発ガイドライン
├── main.py            # エントリーポイント
├── orchestrator.py    # LangGraphマルチエージェント制御
├── core/
│   ├── config.py      # 設定管理
│   ├── memory.py      # メモリ管理（PostgreSQL + セマンティック検索）
│   ├── models.py      # LLMモデル管理
│   ├── obsidian.py    # Obsidian統合
│   ├── mcp.py         # GitHub MCP統合
│   └── semantic_router.py  # セマンティックルーティング
├── agents/
│   ├── spectra.py     # Spectraエージェント（未使用）
│   ├── lynq.py        # LynQエージェント（未使用）
│   └── paz.py         # Pazエージェント（未使用）
├── docker-compose.yml # PostgreSQL環境
└── requirements.txt   # Python依存関係
```

### データフロー
```
ユーザー入力 → LLMルーティング → エージェント選択 → 応答生成
                ↓
            メモリ検索（PostgreSQL + Obsidian）
```

## Discord拡張計画

現在はCLIベースですが、以下の7フェーズでDiscordボット3体に拡張予定：

### Phase 1: 環境セットアップ・Discord.py統合準備
### Phase 2: 共有コアモジュール分離・抽象化  
### Phase 3: Spectraボット実装・テスト
### Phase 4: LynQボット実装・テスト
### Phase 5: Pazボット実装・テスト
### Phase 6: 3ボット同時稼働・統合テスト
### Phase 7: VPS展開・本番運用準備

## パフォーマンス

- **ルーティング精度**: 95% (LLM-driven)
- **起動時間**: 3-5秒 (Obsidian 102ノート読み込み)
- **応答時間**: 1-3秒 (Gemini API + 検索処理)
- **メモリ使用量**: 約500MB (PostgreSQL + LLM推論)

## ライセンス

Private Repository - sito-sikino

## 開発履歴

- **v0.1**: 基本的なLangGraphマルチエージェントシステム
- **v0.2**: メモリ永続化（PostgreSQL + セマンティック検索）
- **v0.3**: LLM-drivenルーティング95%精度達成 + Obsidian完全統合