"""LINE Messaging API 通知モジュール"""

from __future__ import annotations

import requests

from src.config import LINE_CHANNEL_TOKEN, LINE_USER_ID
from src.utils.logging_config import logger

LINE_API_URL = "https://api.line.me/v2/bot/message/push"


class LINENotifier:
    """LINE Messaging APIを使った通知送信"""

    def __init__(
        self,
        channel_token: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self._token = channel_token or LINE_CHANNEL_TOKEN
        self._user_id = user_id or LINE_USER_ID
        if not self._token or not self._user_id:
            logger.warning("LINE認証情報が未設定です")

    def send(self, text: str) -> bool:
        """テキストメッセージを送信

        Args:
            text: 送信テキスト（5000文字まで）

        Returns:
            送信成功: True
        """
        if not self._token or not self._user_id:
            logger.warning("LINE認証情報が未設定のため送信スキップ")
            return False

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        payload = {
            "to": self._user_id,
            "messages": [
                {
                    "type": "text",
                    "text": text[:5000],
                }
            ],
        }

        try:
            resp = requests.post(
                LINE_API_URL,
                headers=headers,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"LINE送信エラー: {e}")
            return False
