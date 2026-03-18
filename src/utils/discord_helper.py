"""Discord Webhook 送信ヘルパー（バックテスト用共通モジュール）"""

from __future__ import annotations

import time

from src.config import DISCORD_WEBHOOK_URL


def send_discord_text(content: str) -> None:
    """Discord Webhook にテキストを送信（2000文字制限対策: 分割送信）"""
    if not DISCORD_WEBHOOK_URL:
        return

    import requests

    try:
        for i in range(0, len(content), 1900):
            chunk = content[i:i + 1900]
            requests.post(
                DISCORD_WEBHOOK_URL,
                json={"content": chunk},
                timeout=10,
            )
            time.sleep(0.5)
    except Exception as e:
        print(f"Discord送信エラー: {e}")
