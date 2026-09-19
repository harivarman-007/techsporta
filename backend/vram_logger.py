"""
Utility: log current NVIDIA GPU VRAM usage via nvidia-smi.
Called after each model loads to verify combined usage stays under ~2 GB.
"""

import subprocess
import logging

logger = logging.getLogger(__name__)


def log_vram(label: str = "") -> dict | None:
    """
    Query nvidia-smi for VRAM usage.

    Returns dict with keys: used_mb, total_mb, free_mb
    Returns None if nvidia-smi is not available.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi returned non-zero: %s", result.stderr)
            return None

        parts = [int(x.strip()) for x in result.stdout.strip().split(",")]
        used_mb, total_mb, free_mb = parts[0], parts[1], parts[2]

        tag = f"[{label}] " if label else ""
        logger.info(
            "%sVRAM — used: %d MB / %d MB  (free: %d MB)",
            tag, used_mb, total_mb, free_mb,
        )
        return {"used_mb": used_mb, "total_mb": total_mb, "free_mb": free_mb}

    except FileNotFoundError:
        logger.warning("nvidia-smi not found — VRAM logging skipped.")
        return None
    except Exception as exc:
        logger.warning("VRAM log failed: %s", exc)
        return None
