#!/usr/bin/env python3
"""
Phase 1 Discord統合テスト

基本的なDiscord統合機能の動作確認
"""

import asyncio
import os
from dotenv import load_dotenv
from src.core.multi_bot_system import MultiDiscordBotSystem
from src.core.bot_config import BotConfig, BotType

# 環境変数を読み込み
load_dotenv()

async def test_discord_system_initialization():
    """Discord統合システムの初期化テスト"""
    try:
        print("🧪 Discord統合システム初期化テスト開始")
        
        # 環境変数確認
        tokens = {
            'spectra': os.getenv("SPECTRA_DISCORD_TOKEN"),
            'lynq': os.getenv("LYNQ_DISCORD_TOKEN"),
            'paz': os.getenv("PAZ_DISCORD_TOKEN"),
            'guild_id': os.getenv("DISCORD_GUILD_ID"),
            'gemini_key': os.getenv("GEMINI_API_KEY")
        }
        
        missing_vars = [k for k, v in tokens.items() if not v]
        if missing_vars:
            print(f"❌ 不足している環境変数: {missing_vars}")
            return False
        
        print("✅ 必要な環境変数確認完了")
        
        # ボット設定作成
        spectra_config = BotConfig(
            name="Spectra",
            bot_type=BotType.SPECTRA,
            token=tokens['spectra'],
            command_prefix="!",
            description="基底エージェント兼一般対話ボット",
            is_primary_receiver=True
        )
        
        lynq_config = BotConfig(
            name="LynQ",
            bot_type=BotType.LYNQ,
            token=tokens['lynq'],
            command_prefix="!",
            description="論理分析専門エージェント"
        )
        
        paz_config = BotConfig(
            name="Paz",
            bot_type=BotType.PAZ,
            token=tokens['paz'],
            command_prefix="!",
            description="創作アイデア専門エージェント"
        )
        
        print("✅ ボット設定作成完了")
        
        # マルチボットシステム初期化
        print("🔄 MultiDiscordBotSystem初期化中...")
        bot_system = MultiDiscordBotSystem(
            spectra_config=spectra_config,
            lynq_config=lynq_config,
            paz_config=paz_config,
            guild_id=int(tokens['guild_id']),
            gemini_api_key=tokens['gemini_key']
        )
        
        print("✅ マルチボットシステム初期化成功")
        
        # ボットインスタンス遅延初期化をテスト
        bot_system._initialize_bots()
        print("✅ ボットインスタンス遅延初期化成功")
        
        # 各ボットの基本プロパティ確認
        bots_info = {
            "Spectra": bot_system.spectra_bot.bot_name,
            "LynQ": bot_system.lynq_bot.bot_name,
            "Paz": bot_system.paz_bot.bot_name
        }
        
        print("📊 ボット情報:")
        for name, bot_name in bots_info.items():
            print(f"   {name}: {bot_name}")
        
        print("🎉 Discord統合システム初期化テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ 初期化テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_bot_config_validation():
    """ボット設定検証テスト"""
    try:
        print("\n🔧 ボット設定検証テスト開始")
        
        # 設定データクラスのテスト
        test_config = BotConfig(
            name="TestBot",
            bot_type=BotType.SPECTRA,
            token="test_token",
            command_prefix="?",
            description="テスト用ボット",
            is_primary_receiver=False
        )
        
        # 設定値確認
        assert test_config.name == "TestBot"
        assert test_config.bot_type == BotType.SPECTRA
        assert test_config.command_prefix == "?"
        assert test_config.is_primary_receiver == False
        
        print("✅ BotConfig データクラス正常動作確認")
        
        # BotType列挙の確認
        bot_types = [BotType.SPECTRA, BotType.LYNQ, BotType.PAZ]
        type_values = [bt.value for bt in bot_types]
        expected_values = ["spectra", "lynq", "paz"]
        
        assert type_values == expected_values
        print("✅ BotType 列挙正常動作確認")
        
        print("🎯 ボット設定検証テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ 設定検証テストエラー: {e}")
        return False

async def main():
    """メインテスト実行"""
    print("=" * 60)
    print("🚀 Phase 1 Discord統合テスト")
    print("=" * 60)
    
    # 設定検証テスト
    config_test_success = await test_bot_config_validation()
    
    # システム初期化テスト
    init_test_success = await test_discord_system_initialization()
    
    print("\n" + "=" * 60)
    print("📋 テスト結果サマリー")
    print("=" * 60)
    print(f"設定検証: {'✅ 成功' if config_test_success else '❌ 失敗'}")
    print(f"システム初期化: {'✅ 成功' if init_test_success else '❌ 失敗'}")
    
    overall_success = config_test_success and init_test_success
    print(f"\n総合判定: {'🎉 Discord統合テスト PASS' if overall_success else '❌ Discord統合テスト FAIL'}")
    
    if overall_success:
        print("\n💡 注意: 実際のDiscord接続テストは手動で実行が必要です")
        print("   - 各BotトークンがDiscord上で有効であること")
        print("   - Guild (サーバー) IDが正しく設定されていること")
        print("   - Botがサーバーに招待されていること")
    
    return overall_success

if __name__ == "__main__":
    asyncio.run(main())