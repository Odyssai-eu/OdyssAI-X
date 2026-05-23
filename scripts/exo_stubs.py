"""Minimal stubs for exo types/logger used by auto_parallel.py.

We only need PipelineShardMetadata for pipeline_auto_parallel and
ModelLoadingResponse for progress reporting. Logger is replaced by stdlib.
"""

import logging
import sys
from dataclasses import dataclass


_logger = logging.getLogger("auto_parallel")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[ap] %(message)s"))
    _logger.addHandler(h)
logger = _logger


@dataclass
class ModelLoadingResponse:
    layers_loaded: int = 0
    total: int = 0


@dataclass
class PipelineShardMetadata:
    """Mirrors exo.shared.types.worker.shards.PipelineShardMetadata fields used by auto_parallel."""

    device_rank: int
    world_size: int
    start_layer: int = 0
    end_layer: int = 0
    immediate_exception: bool = False
    should_timeout: float | None = None
