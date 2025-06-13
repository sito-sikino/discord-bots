"""
ボット設定クラス

循環インポートを避けるため、BotConfigを独立したモジュールに分離
"""

from dataclasses import dataclass
from enum import Enum


class BotType(Enum):
    """ボットタイプ列挙"""
    SPECTRA = "spectra"
    LYNQ = "lynq"
    PAZ = "paz"


@dataclass
class BotConfig:
    """ボット設定情報"""
    name: str
    bot_type: BotType
    token: str
    command_prefix: str = "!"
    description: str = ""
    is_primary_receiver: bool = False