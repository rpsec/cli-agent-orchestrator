"""Unit tests for Gemini CLI provider."""

import re
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.gemini_cli import GeminiCliProvider

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(filename: str) -> str:
    """Load a fixture file and return its contents."""
    with open(FIXTURES_DIR / filename, "r") as f:
        return f.read()


class TestGeminiCliProviderInitialization:
    """Test Gemini CLI provider initialization."""

    @patch("cli_agent_orchestrator.providers.gemini_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.gemini_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_initialize_success(self, mock_tmux, mock_wait_status, mock_wait_shell):
        """Test successful initialization."""
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = True

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        result = provider.initialize()

        assert result is True
        mock_wait_shell.assert_called_once()
        mock_tmux.send_keys.assert_called_once_with(
            "test-session", "window-0", "gemini-cli chat --agent developer"
        )
        mock_wait_status.assert_called_once()

    @patch("cli_agent_orchestrator.providers.gemini_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_initialize_shell_timeout(self, mock_tmux, mock_wait_shell):
        """Test initialization with shell timeout."""
        mock_wait_shell.return_value = False

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")

        with pytest.raises(TimeoutError, match="Shell initialization timed out"):
            provider.initialize()

    @patch("cli_agent_orchestrator.providers.gemini_cli.wait_for_shell")
    @patch("cli_agent_orchestrator.providers.gemini_cli.wait_until_status")
    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_initialize_gemini_cli_timeout(self, mock_tmux, mock_wait_status, mock_wait_shell):
        """Test initialization with Gemini CLI timeout."""
        mock_wait_shell.return_value = True
        mock_wait_status.return_value = False

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")

        with pytest.raises(TimeoutError, match="Gemini CLI initialization timed out"):
            provider.initialize()

    def test_initialization_with_different_agent_profiles(self):
        """Test initialization with various agent profile names."""
        test_profiles = ["developer", "code-reviewer", "test_agent", "agent123"]

        for profile in test_profiles:
            provider = GeminiCliProvider("test1234", "test-session", "window-0", profile)
            assert provider._agent_profile == profile
            # Verify dynamic prompt pattern includes the profile
            assert re.escape(profile) in provider._idle_prompt_pattern


class TestGeminiCliProviderStatusDetection:
    """Test status detection logic."""

    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_get_status_idle(self, mock_tmux):
        """Test IDLE status detection."""
        mock_tmux.get_history.return_value = load_fixture("gemini_cli_idle_output.txt")

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        status = provider.get_status()

        assert status == TerminalStatus.IDLE

    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_get_status_completed(self, mock_tmux):
        """Test COMPLETED status detection."""
        mock_tmux.get_history.return_value = load_fixture("gemini_cli_completed_output.txt")

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        status = provider.get_status()

        assert status == TerminalStatus.COMPLETED

    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_get_status_processing(self, mock_tmux):
        """Test PROCESSING status detection."""
        mock_tmux.get_history.return_value = load_fixture("gemini_cli_processing_output.txt")

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        status = provider.get_status()

        assert status == TerminalStatus.PROCESSING

    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_get_status_error(self, mock_tmux):
        """Test ERROR status detection."""
        mock_tmux.get_history.return_value = load_fixture("gemini_cli_error_output.txt")

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        status = provider.get_status()

        assert status == TerminalStatus.ERROR

    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_get_status_waiting_user_answer(self, mock_tmux):
        """Test WAITING_USER_ANSWER status detection."""
        mock_tmux.get_history.return_value = load_fixture("gemini_cli_permission_output.txt")

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        status = provider.get_status()

        assert status == TerminalStatus.WAITING_USER_ANSWER

    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_get_status_with_tail_lines(self, mock_tmux):
        """Test status detection with tail_lines parameter."""
        mock_tmux.get_history.return_value = load_fixture("gemini_cli_idle_output.txt")

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        status = provider.get_status(tail_lines=50)

        assert status == TerminalStatus.IDLE
        mock_tmux.get_history.assert_called_once_with("test-session", "window-0", tail_lines=50)

    @patch("cli_agent_orchestrator.providers.gemini_cli.tmux_client")
    def test_get_status_empty_output(self, mock_tmux):
        """Test status detection with empty output."""
        mock_tmux.get_history.return_value = ""

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        status = provider.get_status()

        assert status == TerminalStatus.ERROR


class TestGeminiCliProviderMessageExtraction:
    """Test message extraction logic."""

    def test_extract_last_message_from_script(self):
        """Test extracting the last message from terminal output."""
        script_output = load_fixture("gemini_cli_completed_output.txt")

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        message = provider.extract_last_message_from_script(script_output)

        # Verify the message is extracted and cleaned
        assert "Here is a comprehensive response to your query" in message
        assert "This response includes multiple paragraphs" in message
        # Verify ANSI codes are removed
        assert "[38;5;10m" not in message
        assert "[39m" not in message

    def test_extract_last_message_complex_response(self):
        """Test extracting message from complex response."""
        script_output = load_fixture("gemini_cli_complex_response.txt")

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        message = provider.extract_last_message_from_script(script_output)

        # Verify all sections are extracted
        assert "Section 1: Introduction" in message
        assert "Section 2: Technical Details" in message
        assert "Code Example" in message
        assert "Conclusion" in message

    def test_extract_last_message_no_green_arrow(self):
        """Test error when no green arrow pattern is found."""
        script_output = load_fixture("gemini_cli_idle_output.txt")

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")

        with pytest.raises(ValueError, match="No Gemini CLI response found"):
            provider.extract_last_message_from_script(script_output)

    def test_extract_last_message_no_final_prompt(self):
        """Test error when no final prompt is found."""
        script_output = "$ gemini-cli chat --agent developer\n\x1b[38;5;10m> \x1b[39mResponse without final prompt"

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")

        with pytest.raises(ValueError, match="Incomplete Gemini CLI response"):
            provider.extract_last_message_from_script(script_output)

    def test_extract_last_message_empty_response(self):
        """Test error when response is empty."""
        script_output = "$ gemini-cli chat --agent developer\n\x1b[38;5;10m> \x1b[39m\n\n\x1b[36m[developer]\x1b[35m>\x1b[39m "

        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")

        with pytest.raises(ValueError, match="Empty Gemini CLI response"):
            provider.extract_last_message_from_script(script_output)


class TestGeminiCliProviderMethods:
    """Test provider utility methods."""

    def test_get_idle_pattern_for_log(self):
        """Test getting the idle pattern for log file detection."""
        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        pattern = provider.get_idle_pattern_for_log()

        # Should return the IDLE_PROMPT_PATTERN_LOG constant
        assert pattern == r"\x1b\[38;5;13m>\s*\x1b\[39m"

    def test_exit_cli(self):
        """Test getting the exit command."""
        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        exit_command = provider.exit_cli()

        assert exit_command == "/exit"

    def test_cleanup(self):
        """Test cleanup method."""
        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")
        provider._initialized = True

        provider.cleanup()

        assert provider._initialized is False

    def test_provider_properties(self):
        """Test provider properties are set correctly."""
        provider = GeminiCliProvider("test1234", "test-session", "window-0", "developer")

        assert provider.terminal_id == "test1234"
        assert provider.session_name == "test-session"
        assert provider.window_name == "window-0"
        assert provider._agent_profile == "developer"
