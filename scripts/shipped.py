"""Where Datalab's shipped Indra models live -- pinned to one exact revision.

Every script here resolves the models through this one module, so a laptop and a
cluster node are guaranteed to use the same weights: an explicit INDRA_MODEL_DIR
wins; otherwise the local Hugging Face cache of the pinned snapshot; otherwise a
download of exactly that snapshot. A newer Indra release can never be picked up
silently, which would make before/after comparisons measure the release instead
of the fine-tune.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = "datalab-to/indra-ocr"
REVISION = "d7c839c892bfe4b8993373db64c62f154d05d70b"


def shipped_dir() -> Path:
    if os.environ.get("INDRA_MODEL_DIR"):
        return Path(os.environ["INDRA_MODEL_DIR"])
    cached = Path.home() / ".cache/huggingface/hub/models--datalab-to--indra-ocr/snapshots" / REVISION
    if (cached / "rec.onnx").exists():
        return cached
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(REPO, revision=REVISION))


MODEL_DIR = shipped_dir()
