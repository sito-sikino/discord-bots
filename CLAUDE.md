# CLAUDE.md

このファイルは、このリポジトリでコードを扱う際のClaude Code (claude.ai/code) への指針を提供します。

## プロジェクト概要

このプロジェクトは、**LangGraphベース統合プロセス型**で3つの独立したDiscord Botクライアントが協調動作する効率化されたマルチボットLLMシステムです：
- **Spectra Bot**: 基底エージェント兼一般対話ボット（単独受信・ワークフロー起動）
- **LynQ Bot**: 論理分析専門エージェント（LangGraphワークフロー制御）
- **Paz Bot**: 創作アイデア専門エージェント（LangGraphワークフロー制御）

## システムアーキテクチャ

### LangGraphベース統合プロセス型設計パターン
- **単一プロセス管理**: 1つのPythonプロセスで3つの独立したDiscordクライアント + LangGraphワークフロー
- **Discord独立性**: 各ボットが独立したトークン・アイコン・名前・個性を持つ
- **単独受信**: Spectraのみがメッセージを受信し、LangGraphワークフローを起動
- **StateGraph制御**: 明確で可視化されたワークフロー定義による効率的委譲
- **TypedDict状態管理**: 構造化された状態管理とoperator.addによるメッセージ蓄積
- **条件付きルーティング**: add_conditional_edgesによる動的エージェント選択
- **LangChain標準**: ChatGoogleGenerativeAI + 公式パターンによる標準化

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

### 主要依存関係（LangGraph準拠）
- discord.py - Discord botフレームワーク
- langchain - LLMフレームワーク基盤
- langgraph - StateGraphワークフロー管理
- langchain-google-genai - Gemini API統合（ChatGoogleGenerativeAI）
- typing-extensions - TypedDict状態管理
- operator - メッセージ蓄積（operator.add）

**LangGraph最適化**: 公式パターン準拠、StateGraph可視化、条件付きエッジルーティング

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

### LangGraphワークフロー動作フロー

#### Pattern 1: Spectra自己応答（一般対話）
1. **単独受信**: Spectraのみがメッセージを受信
2. **StateGraph起動**: AgentState構築 → analyze_node実行
3. **深層分析**: LLMで意図・文脈を分析（1回目）
4. **ルーティング**: 条件付きエッジで "spectra" 選択
5. **直接応答**: spectra_respond_nodeでLLM応答（2回目）
6. **Discord投稿**: AgentState.messagesから応答取得・投稿

#### Pattern 2: 専門ボット直接委譲（効率化版）
1. **単独受信**: Spectraのみがメッセージを受信
2. **StateGraph起動**: AgentState構築 → analyze_node実行
3. **深層分析**: LLMで意図・文脈を分析（1回目）
4. **ルーティング**: 条件付きエッジで "lynq/paz" 選択
5. **専門応答**: lynq/paz_respond_nodeでLLM応答（2回目）
6. **Discord投稿**: 専門エージェント名義で応答投稿

#### Pattern 3: マルチエージェント協調（複合要求）
1. **単独受信**: Spectraのみがメッセージを受信
2. **StateGraph起動**: AgentState構築 → analyze_node実行
3. **深層分析**: LLMで複合要求を分析（1回目）
4. **ルーティング**: 条件付きエッジで "multi" 選択
5. **協調制御**: multi_coordinate_nodeで調整メッセージ
6. **統合応答**: StateGraph内での順次実行・統合

### LangGraph制御メカニズム
- **StateGraph**: 可視化されたワークフロー定義（Redis不要）
- **TypedDict状態**: 構造化された状態管理とメッセージ蓄積
- **条件付きエッジ**: 動的ルーティングによる最適化
- **LangChain統合**: 公式パターンによる安定性・保守性向上

## 開発ガイドライン

### LangGraphベース実装パターン
メインシステムは以下を実装します：

#### コアコンポーネント
- **MultiDiscordBotSystem**: 3つのDiscordクライアント + LangGraphワークフロー統合管理
- **MultiAgentWorkflow**: StateGraphによるワークフロー定義・実行
- **AgentState (TypedDict)**: 構造化された状態管理
- **ConversationMemory**: ファイルベース履歴管理
- **ChatGoogleGenerativeAI**: 公式Gemini API統合

#### エージェントノード
- **analyze_node**: Spectraによる深層分析・ルーティング判断
- **spectra_respond_node**: 一般対話・ファシリテーション応答
- **lynq_respond_node**: 論理分析専門応答
- **paz_respond_node**: 創作アイデア専門応答
- **multi_coordinate_node**: マルチエージェント協調制御

### LangGraphワークフロー戦略
- **条件付きルーティング**: analyze_nodeの分析結果に基づく自動エージェント選択
- **StateGraph制御**: 明確で追跡可能なワークフロー実行
- **型安全性**: TypedDictによる構造化された状態管理
- **エラー処理**: 統一されたエラーハンドリングとグレースフル縮退

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

### 重要な実装原則（LangGraph版）
- **venv環境必須**: 必ずvenv仮想環境内で開発・実行
- **LangChain準拠**: 公式パターン（StateGraph + TypedDict + ChatGoogleGenerativeAI）
- **Discord独立性**: 3つの独立したBotアカウント維持
- **ワークフロー可視化**: StateGraphによる明確なフロー定義
- **型安全性**: TypedDictによる構造化された状態管理
- **エラー耐性**: 統一されたエラーハンドリングとグレースフル縮退

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