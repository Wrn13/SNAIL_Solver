"""Durable, tailable progress logging for long solver runs.

``print()`` output can sit unflushed under ``srun`` until the job ends. The logger
from ``setup_run_logger`` writes timestamped lines to stdout (captured per job in
``slurm-<jobid>.out``) and optionally to a log file beside the run's outputs.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_run_logger(log_path: Optional[str], name: str) -> logging.Logger:
    """Logger that writes timestamped lines to stdout and, if given, ``log_path``.

    Parameters
    ----------
    log_path : str, optional
        File to append to (parent created if missing). Leave unset for stdout only,
        the right default for concurrent jobs that would interleave into one file.
    name : str
        Logger name; derive it from ``log_path`` so runs writing to different files
        don't share (and duplicate onto) handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            handlers.append(logging.FileHandler(log_path))
        for handler in handlers:
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger
