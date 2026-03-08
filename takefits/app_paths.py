from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths

FALLBACK_APP_DIR_NAME = 'takefits'

def get_app_config_dir() -> Path:
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
