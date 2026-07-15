"""config.example.yaml must stay loadable against the AdGateConfig schema."""

from pathlib import Path

from adgate.config import AdGateConfig, load_config

EXAMPLE = Path(__file__).parent.parent / "config.example.yaml"


def test_example_config_loads():
    # All active keys must validate; a fully-commented example equals defaults.
    assert load_config(EXAMPLE) == AdGateConfig()
