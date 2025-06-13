# CLAUDE.md

このファイルは、このリポジトリでコードを扱う際のClaude Code (claude.ai/code) への指針を提供します。

## プロジェクト概要

このプロジェクトは、**統合プロセス型**で3つの独立したDiscord Botクライアントが協調動作する効率化されたマルチボットLLMシステムです：
- **Spectra Bot**: 基底エージェント兼一般対話ボット（単独受信）
- **LynQ Bot**: 論理分析専門エージェント（シグナル待機）
- **Paz Bot**: 創作アイデア専門エージェント（シグナル待機）

## システムアーキテクチャ

### 統合プロセス型設計パターン
- **単一プロセス管理**: 1つのPythonプロセスで3つの独立したDiscordクライアントを効率管理
- **Discord独立性**: 各ボットが独立したトークン・アイコン・名前・個性を持つ
- **単独受信**: Spectraのみがメッセージを受信し、レート制限を回避
- **効率化委譲**: 中間Discord投稿なしでサブエージェントに直接委譲
- **軽量シグナリング**: AsyncIO標準キューによる内部通信
- **レート制限対応**: 最大2回のシーケンシャルLLMアクセス

### ボットの二重役割・責任範囲
- **Spectra**: 基底エージェント（メタ分析・意思決定） + 一般対話ボット（ファシリテーション）
- **LynQ**: 論理分析専門（問題分解・矛盾指摘・データ分析）
- **Paz**: 創作専門（インスパイアーション・ブレインストーミング・革新的解決）

## 技術スタック

### 基盤技術
- **言語**: Python 3.11+
- **LLM**: Gemini 2.0 Flash API
- **フレームワーク**: Langchain + Langgraph + LangSmith
- **プラットフォーム**: Discord Bot API (discord.py)
- **知識統合**: Obsidian MCP Server (https://github.com/MarkusPfundstein/mcp-obsidian)

### 主要依存関係（軽量化）
- discord.py - Discord botフレームワーク
- langchain - LLMフレームワーク
- langgraph - エージェントワークフロー管理
- langchain-google-genai - Gemini APIクライアント
- asyncio - 非同期処理（Python標準）
- json - データ管理（Python標準）

**最適化ポイント**: Redis不要、データベース不要、追加ライブラリ最小限

### 開発環境
- **仮想環境**: 必ずvenv仮想環境内で開発（`python -m venv venv`）
- **依存関係管理**: requirements.txtまたはpyproject.tomlで管理
- **環境変数**: .envファイルでAPIキー・設定管理（.env.exampleも用意）

### GitHub設定情報（メタ認知用）
- **リポジトリ**: https://github.com/sito-sikino/discord-bots
- **ユーザー名**: sito-sikino
- **メールアドレス**: 213737344+sito-sikino@users.noreply.github.com  
- **ブランチ戦略**: devブランチで開発（mainから派生）
- **LLMモデル**: gemini-2.0-flash（正式版使用）

## データ管理

### 会話履歴
- **構造**: `/data/channels/{channel_id}.json`
- **保持期間**: チャンネルごとに直近20メッセージ
- **形式**: ユーザーメッセージとボット応答を含むJSON
- **アクセス**: 適切なロックを伴うファイルベース共有

### Obsidian統合
- **プロトコル**: Model Context Protocol (MCP)
- **アクセス**: 知識取得のための読み取り専用ボルトアクセス
- **使用方法**: 会話トピックに基づく文脈的知識注入

## 環境設定

### 必要な環境変数
```bash
# ボットトークン
SPECTRA_DISCORD_TOKEN=
LYNQ_DISCORD_TOKEN=
PAZ_DISCORD_TOKEN=

# LLM設定
GEMINI_API_KEY=
GEMINI_API_COORDINATOR_URL=http://localhost:8080

# Discord設定
DISCORD_GUILD_ID=

# 知識ベース
OBSIDIAN_VAULT_PATH=
MCP_SERVER_URL=

# システム設定
LIGHTWEIGHT_DECISION_ENABLED=true
COLLABORATION_MODE=sequential
LLM_ACCESS_TIMEOUT=10

# 効率化設定
DIRECT_DELEGATION_ENABLED=true
INTERMEDIATE_POSTS_DISABLED=true
MAX_LLM_CALLS_PER_REQUEST=2
```

## 効率化された開発ワークフロー

### 3パターン動作フロー

#### Pattern 1: Spectra自己応答（一般対話）
1. **単独受信**: Spectraのみがメッセージを受信
2. **深層分析**: LLMで意図・文脈を分析（1回目）
3. **自己判断**: 一般対話がSpectra最適と判断
4. **直接応答**: Spectraが直接LLM応答（2回目）
5. **Discord投稿**: 即座に応答投稿

#### Pattern 2: 専門ボット直接委謗（効率化版）
1. **単独受信**: Spectraのみがメッセージを受信
2. **深層分析**: LLMで意図・文脈を分析（1回目）
3. **専門判断**: LynQ/Pazが最適と判断
4. **内部シグナル**: AsyncIOキューでサブエージェントに通知（中間投稿なし）
5. **専門応答**: サブエージェントが直接LLM応答（2回目）
6. **Discord投稿**: サブエージェントが応答投稿

#### Pattern 3: マルチエージェント協調（複合要求）
1. **単独受信**: Spectraのみがメッセージを受信
2. **深層分析**: LLMで複合要求を分析（1回目）
3. **協調判断**: 複数ボットが必要と判断
4. **調整投稿**: Spectraが協調開始を通知
5. **順次実行**: LynQ→Paz→Spectraの順で応答
6. **統合応答**: Spectraが最終統合を実行

### 軽量調整メカニズム
- **AsyncIOキュー**: 内部シグナリング（Redis不要）
- **ファイルベースMemory**: 軽量履歴共有
- **レート制限対応**: 最大2回のシーケンシャルLLMアクセス
- **重複回避**: 単一制御点による完全な重複防止

## 開発ガイドライン

### 統合プロセス実装パターン
メインシステムは以下を実装します：

#### コアコンポーネント
- **MultiDiscordBotSystem**: 3つのDiscordクライアント統合管理
- **BotSignalManager**: AsyncIOキューによる内部シグナリング
- **LangGraphエージェント**: Spectraの分析・判断フロー
- **ConversationMemory**: ファイルベース履歴管理
- **ObsidianMCPClient**: 知識ベース統合

#### ボット固有機能
- **Spectra**: 基底エージェント＋一般対話ボット
- **LynQ/Paz**: シグナル待機＋専門応答ボット

### 効率化された応答戦略
- **Spectra直接**: 一般対話・ファシリテーションが最適な場合
- **即時委謗**: 専門性が明確な場合（中間投稿なし）
- **協調委謗**: 複数視点が必要な複合要求（調整投稿あり）

### エラーハンドリング・復旧機構
- **サブエージェント障害**: Spectraのみで縮退運用継続
- **Gemini API制限**: レート制限時の適切な待機・リトライ
- **ファイル競合**: ファイルロック機構で競合解決
- **Discord接続**: 自動再接続・接続状態監視
- **シグナルキュー**: AsyncIOキュー障害時の復旧ロジック

## 実装フェーズ計画

### Phase 1: 基盤システム構築
- プロジェクト初期設定（構造・依存関係・環境変数）
- 統合Discordクライアント実装
- AsyncIOシグナリングシステム
- LangGraph基底エージェント

### Phase 2: LLM統合・専門ボット実装
- Gemini API統合（レート制限対応）
- Spectraエージェント（基底＋一般）
- LynQエージェント（論理分析専門）
- Pazエージェント（創作専門）

### Phase 3: 効率化機能・協調システム
- 3パターン動作実装
- メモリ・履歴管理
- マルチエージェント協調制御

### Phase 4: 高度機能・MCP統合
- Obsidian MCP統合
- エラーハンドリング・復旧
- パフォーマンス最適化

### Phase 5: テスト・品質保証
- 単体テスト（コンポーネント別）
- 統合テスト（3パターン動作）
- 負荷・ストレステスト

### Phase 6: 運用・監視・保守
- ログ・監視システム
- 設定・管理機能
- 拡張性・将来対応

## テスト戦略・品質保証

### 単体テスト
- **Discordクライアント機能**: メッセージ受信・送信・イベントハンドリング
- **LangGraphワークフロー**: ノード実行・エッジ分岐・状態管理
- **シグナリングシステム**: AsyncIOキュー動作・待機ループ
- **メモリ管理**: ファイル読み書き・ロック機構

### 統合テスト
- **3パターン動作**: Spectra直接・専門委謗・マルチ協調
- **マルチエージェント協調**: 順次実行・統合応答
- **Gemini API統合**: レート制限・エラーハンドリング
- **MCP統合**: Obsidian接続・ノート検索・知識注入

### 負荷・ストレステスト
- **連続メッセージ処理**: レート制限下での安定性
- **長時間稼働**: メモリリーク・接続維持
- **障害復旧**: Discord接続切断・API障害からの復旧

### 重要な実装原則
- **venv環境必須**: 必ずvenv仮想環境内で開発・実行
- **レート制限遵守**: 最大2回のシーケンシャルLLMアクセス
- **Discord独立性**: 3つの独立したBotアカウント維持
- **効率化委譲**: Pattern 2で中間Discord投稿なし
- **LangGraph標準**: 追加ライブラリ最小限
- **エラー耐性**: グレースフルな縮退機能

### テスト・保管プロセス
- **段階的テスト**: Discord連携可能になった段階から、各ステップごとにDiscord上で動作テスト実施
- **ユーザー承認**: 各テスト完了後、必ずユーザーのOK確認を取る
- **即座保管**: テスト成功・ユーザーOK後、すぐにGitHubリポジトリにプッシュして保管
- **ブランチ管理**: 各フェーズをfeatureブランチで開発、テスト完了後mainにマージ

### 開発フロー例
```bash
# 1. venv環境準備
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または venv\Scripts\activate  # Windows

# 2. 依存関係インストール
pip install -r requirements.txt

# 3. 環境変数設定
cp .env.example .env
# .envファイルを編集してAPIキー設定

# 4. Discord動作テスト（該当段階から）
python src/main.py

# 5. テスト成功・ユーザーOK後
git add .
git commit -m "Phase X完了: 機能説明"
git push origin feature/phase-x
```