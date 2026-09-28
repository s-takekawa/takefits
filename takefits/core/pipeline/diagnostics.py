"""Shared diagnostics returned by GUI-to-pipeline conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PipelineDiagnostic:
    """One actionable pipeline-conversion diagnostic."""

    severity: str
    code: str
    message: str
    location: str = ""

    def to_dict(self) -> Dict[str, str]:
        payload = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.location:
            payload["location"] = self.location
        return payload
