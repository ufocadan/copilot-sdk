"""
CopilotClient Unit Tests

This file is for unit tests. Where relevant, prefer to add e2e tests in e2e/*.py instead.
"""

import pytest

from copilot import CopilotClient
from copilot.types import ModelInfo
from e2e.testharness import CLI_PATH


class TestModelInfoFromDict:
    """Unit tests for ModelInfo.from_dict robustness with nullable/missing fields."""

    def test_parses_normal_model(self):
        data = {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "capabilities": {
                "supports": {"vision": True, "reasoningEffort": False},
                "limits": {"max_context_window_tokens": 128000},
            },
        }
        model = ModelInfo.from_dict(data)
        assert model.id == "gpt-4o"
        assert model.capabilities.supports.vision is True
        assert model.capabilities.supports.reasoning_effort is False

    def test_parses_model_with_null_vision(self):
        """Models returning vision: null should be parsed as vision=False."""
        data = {
            "id": "gemini-3-flash",
            "name": "Gemini 3 Flash (Preview)",
            "capabilities": {
                "supports": {"vision": None, "reasoningEffort": False},
                "limits": {"max_context_window_tokens": 128000},
            },
        }
        model = ModelInfo.from_dict(data)
        assert model.id == "gemini-3-flash"
        assert model.capabilities.supports.vision is False

    def test_parses_model_with_missing_vision(self):
        """Models omitting the vision key should be parsed as vision=False."""
        data = {
            "id": "grok-code-fast-1",
            "name": "Groke Code Fast 1",
            "capabilities": {
                "supports": {"reasoningEffort": False},
                "limits": {"max_context_window_tokens": 128000},
            },
        }
        model = ModelInfo.from_dict(data)
        assert model.id == "grok-code-fast-1"
        assert model.capabilities.supports.vision is False

    def test_parses_model_with_missing_reasoning_effort(self):
        """Models omitting reasoningEffort should be parsed as reasoning_effort=False."""
        data = {
            "id": "raptor-mini",
            "name": "Raptor mini (Preview)",
            "capabilities": {
                "supports": {"vision": False},
                "limits": {"max_context_window_tokens": 128000},
            },
        }
        model = ModelInfo.from_dict(data)
        assert model.id == "raptor-mini"
        assert model.capabilities.supports.reasoning_effort is False

    def test_parses_model_with_max_output_tokens(self):
        """Models with max_output_tokens in limits should be parsed correctly."""
        data = {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "capabilities": {
                "supports": {"vision": True, "reasoningEffort": False},
                "limits": {"max_context_window_tokens": 128000, "max_output_tokens": 16384},
            },
        }
        model = ModelInfo.from_dict(data)
        assert model.capabilities.limits.max_output_tokens == 16384

    def test_list_comprehension_succeeds_with_mixed_models(self):
        """list_models() should return all models even when some have null/missing fields."""
        models_data = [
            {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "capabilities": {
                    "supports": {"vision": True, "reasoningEffort": False},
                    "limits": {"max_context_window_tokens": 128000},
                },
            },
            {
                "id": "gemini-3-flash",
                "name": "Gemini 3 Flash (Preview)",
                "capabilities": {
                    "supports": {"vision": None},
                    "limits": {"max_context_window_tokens": 100000},
                },
            },
            {
                "id": "grok-code-fast-1",
                "name": "Groke Code Fast 1",
                "capabilities": {
                    "supports": {},
                    "limits": {"max_context_window_tokens": 131072},
                },
            },
        ]
        models = [ModelInfo.from_dict(m) for m in models_data]
        assert len(models) == 3
        assert [m.id for m in models] == ["gpt-4o", "gemini-3-flash", "grok-code-fast-1"]


class TestHandleToolCallRequest:
    @pytest.mark.asyncio
    async def test_returns_failure_when_tool_not_registered(self):
        client = CopilotClient({"cli_path": CLI_PATH})
        await client.start()

        try:
            session = await client.create_session()

            response = await client._handle_tool_call_request(
                {
                    "sessionId": session.session_id,
                    "toolCallId": "123",
                    "toolName": "missing_tool",
                    "arguments": {},
                }
            )

            assert response["result"]["resultType"] == "failure"
            assert response["result"]["error"] == "tool 'missing_tool' not supported"
        finally:
            await client.force_stop()


class TestURLParsing:
    def test_parse_port_only_url(self):
        client = CopilotClient({"cli_url": "8080", "log_level": "error"})
        assert client._actual_port == 8080
        assert client._actual_host == "localhost"
        assert client._is_external_server

    def test_parse_host_port_url(self):
        client = CopilotClient({"cli_url": "127.0.0.1:9000", "log_level": "error"})
        assert client._actual_port == 9000
        assert client._actual_host == "127.0.0.1"
        assert client._is_external_server

    def test_parse_http_url(self):
        client = CopilotClient({"cli_url": "http://localhost:7000", "log_level": "error"})
        assert client._actual_port == 7000
        assert client._actual_host == "localhost"
        assert client._is_external_server

    def test_parse_https_url(self):
        client = CopilotClient({"cli_url": "https://example.com:443", "log_level": "error"})
        assert client._actual_port == 443
        assert client._actual_host == "example.com"
        assert client._is_external_server

    def test_invalid_url_format(self):
        with pytest.raises(ValueError, match="Invalid cli_url format"):
            CopilotClient({"cli_url": "invalid-url", "log_level": "error"})

    def test_invalid_port_too_high(self):
        with pytest.raises(ValueError, match="Invalid port in cli_url"):
            CopilotClient({"cli_url": "localhost:99999", "log_level": "error"})

    def test_invalid_port_zero(self):
        with pytest.raises(ValueError, match="Invalid port in cli_url"):
            CopilotClient({"cli_url": "localhost:0", "log_level": "error"})

    def test_invalid_port_negative(self):
        with pytest.raises(ValueError, match="Invalid port in cli_url"):
            CopilotClient({"cli_url": "localhost:-1", "log_level": "error"})

    def test_cli_url_with_use_stdio(self):
        with pytest.raises(ValueError, match="cli_url is mutually exclusive"):
            CopilotClient({"cli_url": "localhost:8080", "use_stdio": True, "log_level": "error"})

    def test_cli_url_with_cli_path(self):
        with pytest.raises(ValueError, match="cli_url is mutually exclusive"):
            CopilotClient(
                {"cli_url": "localhost:8080", "cli_path": "/path/to/cli", "log_level": "error"}
            )

    def test_use_stdio_false_when_cli_url(self):
        client = CopilotClient({"cli_url": "8080", "log_level": "error"})
        assert not client.options["use_stdio"]

    def test_is_external_server_true(self):
        client = CopilotClient({"cli_url": "localhost:8080", "log_level": "error"})
        assert client._is_external_server


class TestAuthOptions:
    def test_accepts_github_token(self):
        client = CopilotClient(
            {"cli_path": CLI_PATH, "github_token": "gho_test_token", "log_level": "error"}
        )
        assert client.options.get("github_token") == "gho_test_token"

    def test_default_use_logged_in_user_true_without_token(self):
        client = CopilotClient({"cli_path": CLI_PATH, "log_level": "error"})
        assert client.options.get("use_logged_in_user") is True

    def test_default_use_logged_in_user_false_with_token(self):
        client = CopilotClient(
            {"cli_path": CLI_PATH, "github_token": "gho_test_token", "log_level": "error"}
        )
        assert client.options.get("use_logged_in_user") is False

    def test_explicit_use_logged_in_user_true_with_token(self):
        client = CopilotClient(
            {
                "cli_path": CLI_PATH,
                "github_token": "gho_test_token",
                "use_logged_in_user": True,
                "log_level": "error",
            }
        )
        assert client.options.get("use_logged_in_user") is True

    def test_explicit_use_logged_in_user_false_without_token(self):
        client = CopilotClient(
            {"cli_path": CLI_PATH, "use_logged_in_user": False, "log_level": "error"}
        )
        assert client.options.get("use_logged_in_user") is False

    def test_github_token_with_cli_url_raises(self):
        with pytest.raises(
            ValueError, match="github_token and use_logged_in_user cannot be used with cli_url"
        ):
            CopilotClient(
                {
                    "cli_url": "localhost:8080",
                    "github_token": "gho_test_token",
                    "log_level": "error",
                }
            )

    def test_use_logged_in_user_with_cli_url_raises(self):
        with pytest.raises(
            ValueError, match="github_token and use_logged_in_user cannot be used with cli_url"
        ):
            CopilotClient(
                {"cli_url": "localhost:8080", "use_logged_in_user": False, "log_level": "error"}
            )


class TestSessionConfigForwarding:
    @pytest.mark.asyncio
    async def test_create_session_forwards_client_name(self):
        client = CopilotClient({"cli_path": CLI_PATH})
        await client.start()

        try:
            captured = {}
            original_request = client._client.request

            async def mock_request(method, params):
                captured[method] = params
                return await original_request(method, params)

            client._client.request = mock_request
            await client.create_session({"client_name": "my-app"})
            assert captured["session.create"]["clientName"] == "my-app"
        finally:
            await client.force_stop()

    @pytest.mark.asyncio
    async def test_resume_session_forwards_client_name(self):
        client = CopilotClient({"cli_path": CLI_PATH})
        await client.start()

        try:
            session = await client.create_session()

            captured = {}
            original_request = client._client.request

            async def mock_request(method, params):
                captured[method] = params
                return await original_request(method, params)

            client._client.request = mock_request
            await client.resume_session(session.session_id, {"client_name": "my-app"})
            assert captured["session.resume"]["clientName"] == "my-app"
        finally:
            await client.force_stop()
