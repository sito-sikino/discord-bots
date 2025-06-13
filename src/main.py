"""
マルチボットシステム メインエントリーポイント

Phase 1.2: 基盤システム構築
- 統合Discordクライアント実装
- AsyncIOシグナリングシステム
- ファイルベース会話履歴管理
- 3ボット基本動作
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.core.multi_bot_system import MultiDiscordBotSystem, BotConfig
from src.core.signal_manager import BotSignalManager
from src.core.conversation_memory import ConversationMemory


def setup_logging():
    """ログ設定"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # コンソール出力
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # ファイル出力（logsディレクトリが存在する場合）
    logs_dir = project_root / "logs"
    if logs_dir.exists():
        file_handler = logging.FileHandler(
            logs_dir / "discord_bots.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def load_environment():
    """環境変数読み込み"""
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        return True
    else:
        print(f"⚠️  .envファイルが見つかりません: {env_path}")
        print("💡 .env.exampleを参考に.envファイルを作成してください")
        return False


def validate_environment() -> bool:
    """必要な環境変数の検証"""
    required_vars = [
        "SPECTRA_DISCORD_TOKEN",
        "LYNQ_DISCORD_TOKEN", 
        "PAZ_DISCORD_TOKEN",
        "DISCORD_GUILD_ID",
        "GEMINI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ 必要な環境変数が設定されていません: {', '.join(missing_vars)}")
        return False
    
    return True


def create_bot_configs() -> tuple:
    """ボット設定作成"""
    spectra_config = BotConfig(
        token=os.getenv("SPECTRA_DISCORD_TOKEN"),
        name="Spectra",
        description="基底エージェント兼一般対話ボット（単独受信）",
        command_prefix="!s"
    )
    
    lynq_config = BotConfig(
        token=os.getenv("LYNQ_DISCORD_TOKEN"),
        name="LynQ",
        description="論理分析専門エージェント（シグナル待機）",
        command_prefix="!l"
    )
    
    paz_config = BotConfig(
        token=os.getenv("PAZ_DISCORD_TOKEN"),
        name="Paz",
        description="創作アイデア専門エージェント（シグナル待機）",
        command_prefix="!p"
    )
    
    return spectra_config, lynq_config, paz_config


async def main():
    """メイン実行関数"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 マルチボットシステムを開始...")
        
        # 環境変数読み込み・検証
        if not load_environment():
            return 1
        
        if not validate_environment():
            return 1
        
        # ボット設定作成
        spectra_config, lynq_config, paz_config = create_bot_configs()
        guild_id = int(os.getenv("DISCORD_GUILD_ID"))
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # マルチボットシステム初期化
        multi_bot_system = MultiDiscordBotSystem(
            spectra_config=spectra_config,
            lynq_config=lynq_config,
            paz_config=paz_config,
            guild_id=guild_id,
            gemini_api_key=gemini_api_key
        )
        
        logger.info("📊 システム構成:")
        logger.info(f"  🌟 Spectra: {spectra_config.description}")
        logger.info(f"  🔍 LynQ: {lynq_config.description}")  
        logger.info(f"  🎨 Paz: {paz_config.description}")
        logger.info(f"  🏠 Guild ID: {guild_id}")
        
        # システム起動
        logger.info("⏳ 全ボットを起動中...")
        await multi_bot_system.start_all_bots()
        
    except KeyboardInterrupt:
        logger.info("🛑 ユーザーによる停止要求")
        if 'multi_bot_system' in locals():
            await multi_bot_system.shutdown()
        return 0
        
    except Exception as e:
        logger.error(f"💥 システムエラー: {e}", exc_info=True)
        if 'multi_bot_system' in locals():
            await multi_bot_system.shutdown()
        return 1


def run_system():
    """システム実行（エントリーポイント）"""
    # ログ設定
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("🤖 Discord Multi-Bot LLM System")
    logger.info("Phase 1.2: 基盤システム構築")
    logger.info("=" * 60)
    
    # Python版本確認
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8以上が必要です")
        return 1
    
    # 仮想環境確認（推奨）
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.info("✅ 仮想環境で実行中")
    else:
        logger.warning("⚠️  仮想環境での実行を推奨します")
    
    # システム実行
    try:
        return asyncio.run(main())
    except Exception as e:
        logger.error(f"💥 システム起動失敗: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    """直接実行時のエントリーポイント"""
    exit_code = run_system()
    sys.exit(exit_code)