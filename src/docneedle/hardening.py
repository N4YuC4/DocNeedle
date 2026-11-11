"""Kernel process security and anti-dump hardening for DocNeedle.

Enforces zero-disk leakage by preventing memory dumping (core dumps)
and ptrace memory inspection via Linux PR_SET_DUMPABLE.
"""

import ctypes
import logging
import os
import sys

logger = logging.getLogger(__name__)

PR_SET_DUMPABLE = 4
PR_GET_DUMPABLE = 3


def harden_process() -> None:
    """Disable core dumping and ptrace memory inspection via Linux PR_SET_DUMPABLE.

    Strict fail-fast policy: If running on Linux and dumpable flag cannot be cleared,
    raise OSError immediately, refusing to start in an insecure memory state.

    Bypass for development/debugging: Set DOCNEEDLE_DEV_ALLOW_DUMPABLE=1 in environment.
    """
    if sys.platform != "linux":
        logger.debug("Kernel hardening skipped: platform '%s' is not Linux", sys.platform)
        return

    if os.environ.get("DOCNEEDLE_DEV_ALLOW_DUMPABLE", "0").lower() in ("1", "true", "yes"):
        logger.info("Kernel process hardening bypassed by DOCNEEDLE_DEV_ALLOW_DUMPABLE=1")
        return

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        ret = libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
        if ret != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"libc.prctl(PR_SET_DUMPABLE, 0) failed with errno {errno}")

        # Verify dumpable status was cleared
        dumpable = libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0)
        if dumpable != 0:
            raise RuntimeError(
                f"PR_SET_DUMPABLE verification failed; PR_GET_DUMPABLE returned {dumpable}"
            )

        logger.info(
            "--> [Kernel Hardening] PR_SET_DUMPABLE=0 successfully active (core dumping disabled)"
        )
    except Exception as e:
        logger.critical("Failed to harden process against core dumping: %s", e)
        raise


def is_process_dumpable() -> bool:
    """Check if current Linux process allows core dumping via PR_GET_DUMPABLE."""
    if sys.platform != "linux":
        return True
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0
    except Exception:
        return True
