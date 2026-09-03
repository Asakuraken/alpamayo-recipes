"""Navigation annotations on top of the deferred PAI GOP dataset."""

from __future__ import annotations

import json
from typing import Any

from alpamayo.data.nvc_pai import NvcPAIDataset


class NvcPAIDatasetWithNav(NvcPAIDataset):
    def __init__(self, annotations_path: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        with open(annotations_path) as stream:
            entries = json.load(stream)
        allowed = set(self.clip_ids)
        self._samples = [entry for entry in entries if entry["clip_id"] in allowed]
        if not self._samples:
            raise ValueError("No navigation annotations match configured PAI chunks")
        self.clip_ids = [entry["clip_id"] for entry in self._samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        entry = self._samples[idx]
        sample = self._get_sample(entry["clip_id"], int(entry["t0_relative"]))
        sample["nav_text"] = entry["nav_text"]
        return sample
