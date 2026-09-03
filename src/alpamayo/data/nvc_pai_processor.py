"""Main-process materialization for deferred PAI GOP samples."""

from __future__ import annotations

from typing import Any, Callable

import torch

from alpamayo.data.nvc_gop import NVC_GOP_REQUEST_KEY


def nvc_pai_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep GOP references out of PyTorch's tensor collation path."""

    return {
        "inputs": {
            "_nvc_samples": samples,
            NVC_GOP_REQUEST_KEY: [sample[NVC_GOP_REQUEST_KEY] for sample in samples],
        }
    }


class NvcPAIBatchProcessor:
    """Turn NVDEC output back into the ordinary Alpamayo SFT batch format."""

    def __init__(self, dataset: Any, collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]]):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def collate_decoded_vlm_inputs(
        self, *, decoded_samples: list[torch.Tensor], languages: list[str], device: torch.device
    ) -> dict[str, Any]:
        # ``languages`` is a generic prefetcher argument; PAI's labels already
        # live in the deferred samples and therefore it is intentionally unused.
        del languages, device
        raise RuntimeError(
            "NvcPAIBatchProcessor requires materialize_batch(), which has access "
            "to the deferred PAI samples."
        )

    def materialize_batch(self, batch: dict[str, Any], decoded_samples: list[torch.Tensor]) -> dict[str, Any]:
        inputs = batch["inputs"]
        samples = inputs.pop("_nvc_samples")
        requests = inputs.pop(NVC_GOP_REQUEST_KEY)
        if len(samples) != len(decoded_samples):
            raise RuntimeError("Decoded sample count does not match PAI batch")
        ready: list[dict[str, Any]] = []
        for sample, request, frames in zip(samples, requests, decoded_samples):
            n_cam, n_frames = len(request.camera_features), len(request.frame_ids[0])
            if frames.shape[0] != n_cam * n_frames:
                raise RuntimeError("Unexpected NVDEC frame count for PAI sample")
            # Reference prefetcher emits time-major [T*V,C,H,W]; PAI/Qwen uses
            # [V,T,C,H,W].  ``float`` creates owned storage after NVDEC's pool.
            image_frames = frames.reshape(n_frames, n_cam, *frames.shape[1:]).permute(1, 0, 2, 3, 4)
            item = dict(sample)
            item.pop(NVC_GOP_REQUEST_KEY, None)
            item["image_frames"] = image_frames
            item["tokenized_data"] = self.dataset.vla_preprocess_func(data=item)
            # VLA preprocessing has converted the NVDEC frames into the model
            # inputs in ``tokenized_data``; raw decoded frames are no longer used.
            item.pop("image_frames", None)
            ready.append(item)
        return self.collate_fn(ready)
