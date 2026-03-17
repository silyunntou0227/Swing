"""Discord Webhook 通知モジュール"""

from __future__ import annotations

import time

import requests

from src.config import DISCORD_WEBHOOK_URL
from src.utils.logging_config import logger


class DiscordNotifier:
    """Discord Webhookを使った通知送信"""

    RATE_LIMIT_DELAY = 1.0  # Discord rate limit対策（秒）

    def __init__(self, webhook_url: str | None = None) -> None:
        self._url = webhook_url or DISCORD_WEBHOOK_URL
        if not self._url:
            logger.warning("DISCORD_WEBHOOK_URL が未設定です")

    def send(self, content: str | None = None, embed: dict | None = None) -> bool:
        """メッセージを送信

        Args:
            content: テキストメッセージ（2000文字まで）
            embed: Discord Embed辞書

        Returns:
            送信成功: True
        """
        if not self._url:
            logger.warning("Discord Webhook URLが未設定のため送信スキップ")
            return False

        payload = {}
        if content:
            # Discord は2000文字制限
            payload["content"] = content[:2000]
        if embed:
            payload["embeds"] = [embed]

        if not payload:
            return False

        try:
            resp = requests.post(
                self._url,
                json=payload,
                timeout=10,
            )
            if resp.status_code == 429:
                # Rate limited — wait and retry once
                retry_after = resp.json().get("retry_after", 2)
                logger.warning(f"Discord rate limited, {retry_after}秒待機...")
                time.sleep(retry_after)
                resp = requests.post(self._url, json=payload, timeout=10)

            resp.raise_for_status()
            time.sleep(self.RATE_LIMIT_DELAY)
            return True

        except requests.RequestException as e:
            logger.error(f"Discord送信エラー: {e}")
            return False

    def send_embed(
        self,
        title: str,
        description: str,
        color: int = 0x00FF00,
        fields: list[dict] | None = None,
    ) -> bool:
        """Embed形式でメッセージを送信"""
        embed = {
            "title": title,
            "description": description,
            "color": color,
        }
        if fields:
            embed["fields"] = fields
        return self.send(embed=embed)

    def send_error(self, error_message: str) -> bool:
        """エラー通知を送信"""
        return self.send_embed(
            title="スキャンエラー",
            description=f"```\n{error_message[:1800]}\n```",
            color=0xFF0000,
        )
