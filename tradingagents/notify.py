"""Best-effort notifications through Hermes (``hermes send``): never raises, never blocks an order.

Targets come from ``TRADINGAGENTS_NOTIFY_TO`` (comma-separated ``hermes send --to`` values, e.g.
``email,discord``). Unset means no notification.
"""

from __future__ import annotations

import os
import subprocess


def notify(subject: str, body: str = "") -> list[str]:
    """Send to every configured target; returns the targets that accepted the message."""
    targets = [t.strip() for t in os.environ.get("TRADINGAGENTS_NOTIFY_TO", "").split(",") if t.strip()]
    delivered = []
    for target in targets:
        try:
            proc = subprocess.run(["hermes", "send", "--to", target, "--subject", subject, "--quiet", body or subject],
                                  capture_output=True, text=True, timeout=60)
            if proc.returncode == 0:
                delivered.append(target)
        except (OSError, subprocess.SubprocessError):
            continue
    return delivered
