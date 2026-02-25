"""
Configuration and secrets management.
Uses Infisical for production, falls back to .env for local development.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@dataclass
class SleeperConfig:
    """Sleeper API configuration."""
    base_url: str = "https://api.sleeper.app/v1"
    dynasty_league_id: str = ""
    username: str = ""
    rate_limit_per_min: int = 60


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///nfl_dynasty.db"
    echo: bool = False


@dataclass
class NFLFantasyConfig:
    """NFL Fantasy app scraping config (optional - Phase 2)."""
    email: str = ""
    password: str = ""


@dataclass
class AppConfig:
    """Main application configuration."""
    sleeper: SleeperConfig = field(default_factory=SleeperConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    nfl_fantasy: NFLFantasyConfig = field(default_factory=NFLFantasyConfig)
    debug: bool = False
    cache_dir: Path = Path("data_cache")

    def __post_init__(self):
        self.cache_dir.mkdir(exist_ok=True)


def load_config() -> AppConfig:
    """Load configuration from Infisical or environment variables."""
    config = AppConfig()

    # Try Infisical first, fall back to env vars
    try:
        from infisical_client import ClientSettings, InfisicalClient
        infisical = InfisicalClient(ClientSettings(
            client_id=os.getenv("INFISICAL_CLIENT_ID", ""),
            client_secret=os.getenv("INFISICAL_CLIENT_SECRET", ""),
        ))
        project_id = os.getenv("INFISICAL_PROJECT_ID", "")
        env = os.getenv("INFISICAL_ENV", "dev")

        config.sleeper.dynasty_league_id = infisical.getSecret(
            secret_name="SLEEPER_LEAGUE_ID",
            project_id=project_id,
            environment=env,
        ).secret_value
        config.sleeper.username = infisical.getSecret(
            secret_name="SLEEPER_USERNAME",
            project_id=project_id,
            environment=env,
        ).secret_value
        logger.info("Loaded secrets from Infisical")
    except Exception:
        # Fall back to .env
        config.sleeper.dynasty_league_id = os.getenv(
            "SLEEPER_LEAGUE_ID", "1311316930342195200"
        )
        config.sleeper.username = os.getenv("SLEEPER_USERNAME", "")
        logger.info("Using .env for configuration")

    config.debug = os.getenv("DEBUG", "false").lower() == "true"
    return config
