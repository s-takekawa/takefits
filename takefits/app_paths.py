from __future__ import annotations

from pathlib import Path

FALLBACK_APP_DIR_NAME = 'takefits'


def get_app_config_dir() -> Path:
    # Keep importing configuration/schema helpers headless.  Qt is only needed
    # when a caller actually resolves the platform-specific application path.
    from PySide6.QtCore import QStandardPaths

    path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    if path:
        return Path(path)
    return Path.home() / '.config' / FALLBACK_APP_DIR_NAME

def ensure_app_config_dir() -> Path:
    path = get_app_config_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path

def app_config_path(filename: str) -> str:
    return str(ensure_app_config_dir() / filename)

def repository_cli_runner() -> Path:
    """The source checkout's pipeline runner, ``cli/run.py`` beside the package."""
    return Path(__file__).resolve().parents[1] / 'cli' / 'run.py'

def pipeline_runner_available() -> bool:
    """Whether exported pipeline manifests can be run from this installation.

    Only a source checkout carries ``cli/run.py``; installed releases do not,
    so GUI pipeline export stays hidden there until the CLI is packaged.
    """
    return repository_cli_runner().is_file()
