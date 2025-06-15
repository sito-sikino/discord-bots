#!/usr/bin/env python3
"""
LangGraph CLIチャットボット - シンプル版
設定とエントリポイントを統合
"""
import sys
import os
from typing import Optional
from core.config import config
from bot import Bot


def main():
    """メイン関数"""
    bot = Bot(config)
    bot.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n終了します。")
        sys.exit(0)