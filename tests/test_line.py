"""LINE 通知モジュールのユニットテスト"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from src.notify.line import (
    LINENotifier,
    LINE_BROADCAST_URL,
    LINE_PUSH_URL,
    verify_webhook_signature,
)


# ---------- ヘルパー ----------


def _mock_response(status_code: int = 200, request_id: str = "req-123", text: str = ""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"x-line-request-id": request_id}
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


# ---------- Broadcast 送信（デフォルト） ----------


class TestLINENotifierBroadcast:
    def setup_method(self):
        self.notifier = LINENotifier(channel_token="test-token", user_id="test-user")

    @patch("src.notify.line.requests.post")
    def test_send_broadcasts_to_all(self, mock_post):
        """send() は Broadcast API を使い、payload に 'to' が無い"""
        mock_post.return_value = _mock_response(200)
        assert self.notifier.send("hello") is True
        mock_post.assert_called_once()

        # Broadcast URL を使っていること
        call_args = mock_post.call_args
        assert call_args.args[0] == LINE_BROADCAST_URL

        # payload に "to" が無い（全員配信）
        payload = call_args.kwargs["json"]
        assert "to" not in payload
        assert payload["messages"][0]["type"] == "text"
        assert payload["messages"][0]["text"] == "hello"

    @patch("src.notify.line.requests.post")
    def test_send_truncates_to_5000(self, mock_post):
        mock_post.return_value = _mock_response(200)
        long_text = "a" * 6000
        self.notifier.send(long_text)
        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["messages"][0]["text"]) == 5000

    def test_send_skips_without_token(self):
        notifier = LINENotifier(channel_token="", user_id="")
        assert notifier.send("hello") is False

    @patch("src.notify.line.requests.post")
    def test_send_client_error_no_retry(self, mock_post):
        """400 エラーはリトライしない"""
        mock_post.return_value = _mock_response(400, text="Bad Request")
        assert self.notifier.send("hello") is False
        assert mock_post.call_count == 1

    @patch("src.notify.line.time.sleep")
    @patch("src.notify.line.requests.post")
    def test_send_429_retries_with_backoff(self, mock_post, mock_sleep):
        """429 レート制限で指数バックオフリトライ"""
        mock_post.side_effect = [
            _mock_response(429),
            _mock_response(429),
            _mock_response(200),
        ]
        assert self.notifier.send("hello") is True
        assert mock_post.call_count == 3
        assert mock_sleep.call_args_list[0].args[0] == 2
        assert mock_sleep.call_args_list[1].args[0] == 4

    @patch("src.notify.line.time.sleep")
    @patch("src.notify.line.requests.post")
    def test_send_500_retries_and_fails(self, mock_post, mock_sleep):
        """5xx で最大リトライ後に失敗"""
        mock_post.return_value = _mock_response(500)
        assert self.notifier.send("hello") is False
        assert mock_post.call_count == 4

    @patch("src.notify.line.time.sleep")
    @patch("src.notify.line.requests.post")
    def test_send_network_error_retries(self, mock_post, mock_sleep):
        """ネットワークエラーでもリトライ"""
        import requests as req

        mock_post.side_effect = [
            req.ConnectionError("connection lost"),
            _mock_response(200),
        ]
        assert self.notifier.send("hello") is True
        assert mock_post.call_count == 2


# ---------- Push 送信（特定ユーザー） ----------


class TestLINENotifierPush:
    def setup_method(self):
        self.notifier = LINENotifier(channel_token="test-token", user_id="test-user")

    @patch("src.notify.line.requests.post")
    def test_send_to_user_uses_push_api(self, mock_post):
        """send_to_user() は Push API を使い、payload に 'to' がある"""
        mock_post.return_value = _mock_response(200)
        assert self.notifier.send_to_user("hello") is True

        call_args = mock_post.call_args
        assert call_args.args[0] == LINE_PUSH_URL

        payload = call_args.kwargs["json"]
        assert payload["to"] == "test-user"
        assert payload["messages"][0]["text"] == "hello"

    @patch("src.notify.line.requests.post")
    def test_send_to_specific_user(self, mock_post):
        """明示的にユーザーIDを指定"""
        mock_post.return_value = _mock_response(200)
        assert self.notifier.send_to_user("hello", user_id="U999") is True

        payload = mock_post.call_args.kwargs["json"]
        assert payload["to"] == "U999"

    def test_send_to_user_skips_without_credentials(self):
        notifier = LINENotifier(channel_token="", user_id="")
        assert notifier.send_to_user("hello") is False


# ---------- Flex Message ----------


class TestLINENotifierFlex:
    def setup_method(self):
        self.notifier = LINENotifier(channel_token="test-token", user_id="test-user")

    @patch("src.notify.line.requests.post")
    def test_send_flex_broadcasts(self, mock_post):
        mock_post.return_value = _mock_response(200)
        contents = {"type": "bubble", "body": {"type": "box", "layout": "vertical", "contents": []}}
        assert self.notifier.send_flex("alt text", contents) is True

        call_args = mock_post.call_args
        assert call_args.args[0] == LINE_BROADCAST_URL

        payload = call_args.kwargs["json"]
        assert "to" not in payload
        assert payload["messages"][0]["type"] == "flex"
        assert payload["messages"][0]["altText"] == "alt text"

    @patch("src.notify.line.requests.post")
    def test_send_flex_truncates_alt_text(self, mock_post):
        mock_post.return_value = _mock_response(200)
        long_alt = "x" * 500
        self.notifier.send_flex(long_alt, {"type": "bubble"})
        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["messages"][0]["altText"]) == 400


# ---------- マルチメッセージ ----------


class TestLINENotifierMultiMessage:
    def setup_method(self):
        self.notifier = LINENotifier(channel_token="test-token", user_id="test-user")

    @patch("src.notify.line.requests.post")
    def test_send_messages_empty(self, mock_post):
        assert self.notifier.send_messages([]) is True
        mock_post.assert_not_called()

    @patch("src.notify.line.requests.post")
    def test_send_messages_within_limit(self, mock_post):
        mock_post.return_value = _mock_response(200)
        msgs = [{"type": "text", "text": f"msg{i}"} for i in range(3)]
        assert self.notifier.send_messages(msgs) is True
        assert mock_post.call_count == 1

    @patch("src.notify.line.requests.post")
    def test_send_messages_splits_at_5(self, mock_post):
        """6件のメッセージ → 5+1 で2リクエスト"""
        mock_post.return_value = _mock_response(200)
        msgs = [{"type": "text", "text": f"msg{i}"} for i in range(6)]
        assert self.notifier.send_messages(msgs) is True
        assert mock_post.call_count == 2
        first_payload = mock_post.call_args_list[0].kwargs["json"]
        assert len(first_payload["messages"]) == 5
        second_payload = mock_post.call_args_list[1].kwargs["json"]
        assert len(second_payload["messages"]) == 1

    @patch("src.notify.line.requests.post")
    def test_send_messages_stops_on_failure(self, mock_post):
        """最初のバッチが失敗したら中断"""
        mock_post.return_value = _mock_response(400, text="Bad Request")
        msgs = [{"type": "text", "text": f"msg{i}"} for i in range(7)]
        assert self.notifier.send_messages(msgs) is False
        assert mock_post.call_count == 1


# ---------- Webhook 署名検証 ----------


class TestVerifyWebhookSignature:
    def test_valid_signature(self):
        secret = "test-channel-secret"
        body = b'{"events":[]}'
        import hashlib
        import hmac as _hmac
        import base64

        digest = _hmac.new(
            secret.encode("utf-8"), body, hashlib.sha256
        ).digest()
        valid_sig = base64.b64encode(digest).decode("utf-8")

        assert verify_webhook_signature(secret, body, valid_sig) is True

    def test_invalid_signature(self):
        assert verify_webhook_signature(
            "secret", b"body", "invalid-signature"
        ) is False

    def test_tampered_body(self):
        secret = "test-secret"
        original_body = b'{"events":[{"type":"message"}]}'
        import hashlib
        import hmac as _hmac
        import base64

        digest = _hmac.new(
            secret.encode("utf-8"), original_body, hashlib.sha256
        ).digest()
        valid_sig = base64.b64encode(digest).decode("utf-8")

        tampered = b'{"events":[{"type":"follow"}]}'
        assert verify_webhook_signature(secret, tampered, valid_sig) is False
