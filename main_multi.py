#!/usr/bin/env python3
"""
Multi-Agent LangGraph CLI Chatbot Entry Point
Spectra/LynQ/Paz マルチエージェントシステム
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Multi-agent system main entry point"""
    try:
        # Import after env loading
        from orchestrator import MultiAgentOrchestrator
        
        # Initialize and run multi-agent system
        orchestrator = MultiAgentOrchestrator()
        orchestrator.run()
        
    except KeyboardInterrupt:
        print("\n\n終了します。")
        sys.exit(0)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()