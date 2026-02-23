import os
from dataclasses import dataclass, field
from typing import Optional

LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


@dataclass(frozen=True)
class DatabaseConfig:
    """PostgreSQL database configuration derived from DATABASE_URL."""

    url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))

    @property
    def is_configured(self) -> bool:
        """Return True when a database URL is available."""
        return bool(self.url)


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    app_name: str = "DataSciencePlatform"
    version: str = "0.1.0"
    debug: bool = field(
        default_factory=lambda: os.environ.get("DEBUG", "false").lower() == "true"
    )
    upload_dir: str = field(
        default_factory=lambda: os.environ.get("UPLOAD_DIR", "data/uploads")
    )
    model_dir: str = field(
        default_factory=lambda: os.environ.get("MODEL_DIR", "models")
    )
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


_settings: Optional[AppConfig] = None


def get_settings() -> AppConfig:
    """Return a cached singleton of the application settings.

    Returns:
        AppConfig: The application configuration instance.
    """
    global _settings
    if _settings is None:
        _settings = AppConfig()
    return _settings
