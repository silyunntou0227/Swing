"""LINE Messaging API 通知モジュール

報告書に基づく改善:
- 指数バックオフによるリトライ（429/5xx 対応）
- x-line-request-id のログ記録
- Flex Message 対応（リッチ通知）
- マルチメッセージ送信（1リクエストで最大5件）
- Webhook 署名検証ユーティリティ（将来の受信用）
"""

from __future__ import annotations

import hashlib
import hmac
import base64
import time
from typing import Any

import requests

from src.config import LINE_CHANNEL_TOKEN, LINE_USER_ID
from src.utils.logging_config import logger

# LINE Messaging API エンドポイント
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_MULTICAST_URL = "https://api.line.me/v2/bot/message/multicast"

# 旧エンドポイント名の互換性維持
LINE_API_URL = LINE_PUSH_URL

# リトライ設定
MAX_RETRIES = 3
BACKOFF_BASE = 2  # 指数バックオフの基数（秒）


class LINENotifier:
    """LINE Messaging APIを使った通知送信

    機能:
    - テキスト / Flex Message のプッシュ送信
    - 429 Too Many Requests / 5xx エラーの自動リトライ（指数バックオフ）
    - x-line-request-id によるリクエスト追跡ログ
    - 1リクエストで最大5件のメッセージ送信
    """

    def __init__(
        self,
        channel_token: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self._token = channel_token or LINE_CHANNEL_TOKEN
        self._user_id = user_id or LINE_USER_ID
        if not self._token or not self._user_id:
            logger.warning("LINE認証情報が未設定です")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, text: str) -> bool:
        """テキストメッセージを送信

        Args:
            text: 送信テキスト（5000文字まで）

        Returns:
            送信成功: True
        """
        message = {"type": "text", "text": text[:5000]}
        return self._push_messages([message])

    def send_flex(self, alt_text: str, contents: dict[str, Any]) -> bool:
        """Flex Message を送信

        Args:
            alt_text: 通知プレビュー用テキスト（400文字まで）
            contents: Flex Message の contents オブジェクト（bubble or carousel）

        Returns:
            送信成功: True
        """
        message = {
            "type": "flex",
            "altText": alt_text[:400],
            "contents": contents,
        }
        return self._push_messages([message])

    def send_messages(self, messages: list[dict[str, Any]]) -> bool:
        """複数メッセージを一括送信（最大5件）

        Args:
            messages: LINE message object のリスト

        Returns:
            送信成功: True
        """
        if not messages:
            return True
        # LINE API は 1 リクエスト最大 5 メッセージ
        for i in range(0, len(messages), 5):
            chunk = messages[i : i + 5]
            if not self._push_messages(chunk):
                return False
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _push_messages(self, messages: list[dict[str, Any]]) -> bool:
        """メッセージ送信の共通処理（リトライ付き）"""
        if not self._token or not self._user_id:
            logger.warning("LINE認証情報が未設定のため送信スキップ")
            return False

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        payload = {
            "to": self._user_id,
            "messages": messages,
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    LINE_PUSH_URL,
                    headers=headers,
                    json=payload,
                    timeout=10,
                )

                # x-line-request-id をログに記録（トラブルシュート用）
                request_id = resp.headers.get("x-line-request-id", "N/A")

                if resp.status_code == 200:
                    logger.debug(f"LINE送信成功 (request-id: {request_id})")
                    return True

                # 429 (レート制限) または 5xx (サーバーエラー) → リトライ
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < MAX_RETRIES:
                        wait = BACKOFF_BASE ** (attempt + 1)
                        logger.warning(
                            f"LINE API {resp.status_code} "
                            f"(request-id: {request_id}) "
                            f"— {wait}秒後にリトライ ({attempt + 1}/{MAX_RETRIES})"
                        )
                        time.sleep(wait)
                        continue
                    else:
                        logger.error(
                            f"LINE送信失敗（リトライ上限）: {resp.status_code} "
                            f"(request-id: {request_id})"
                        )
                        return False

                # その他のエラー（400, 401, 403 等）→ リトライしない
                logger.error(
                    f"LINE送信エラー: {resp.status_code} {resp.text} "
                    f"(request-id: {request_id})"
                )
                return False

            except requests.RequestException as e:
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        f"LINE通信エラー: {e} — {wait}秒後にリトライ "
                        f"({attempt + 1}/{MAX_RETRIES})"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"LINE送信エラー（リトライ上限）: {e}")
                return False

        return False  # pragma: no cover


# ============================================================
# Webhook 署名検証ユーティリティ
# ============================================================


def verify_webhook_signature(
    channel_secret: str,
    raw_body: bytes,
    signature: str,
) -> bool:
    """Webhook リクエストの署名を検証する

    LINE プラットフォームからのリクエストが改ざんされていないことを
    HMAC-SHA256 で検証する。タイミング攻撃を防ぐため
    hmac.compare_digest を使用。

    Args:
        channel_secret: チャネルシークレット
        raw_body: リクエストボディの生バイト列（JSON デコード前）
        signature: x-line-signature ヘッダーの値

    Returns:
        署名が一致すれば True
    """
    hash_value = hmac.new(
        channel_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).digest()
    expected = base64.b64encode(hash_value).decode("utf-8")
    return hmac.compare_digest(expected, signature)
