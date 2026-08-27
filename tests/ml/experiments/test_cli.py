import subprocess
import sys


def test_cli_invalid_config_returns_non_zero(tmp_path):
    invalid_config_path = tmp_path / "invalid.json"
    invalid_config_path.write_text('{"invalid": "config"}')
    
    result = subprocess.run(
        [sys.executable, "-m", "ml.experiments", str(invalid_config_path)],
        capture_output=True,
        text=True,
        check=False
    )
    assert result.returncode != 0
    assert "Failed to parse or validate configuration" in result.stderr


def test_cli_missing_config_returns_non_zero():
    result = subprocess.run(
        [sys.executable, "-m", "ml.experiments", "nonexistent.json"],
        capture_output=True,
        text=True,
        check=False
    )
    assert result.returncode != 0
    assert "Configuration file not found" in result.stderr

