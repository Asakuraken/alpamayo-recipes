"""ACCV-Lab deferred GOP primitives for Alpamayo training.

The PAI training recipe imports these primitives in its worker dataset and
main-process prefetcher.  Keeping the request object dependency-free makes it
safe to send through a PyTorch DataLoader queue: compressed GOP bytes never
enter that queue; only shared-memory references do.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import ctypes
import os
import struct
from typing import Any

import numpy as np


NVC_GOP_REQUEST_KEY = "_nvc_gop_request"


def own_serialized_gop_bundle(data: Any, *, context: str) -> np.ndarray:
    """Return a stable, validated copy of one ACCV serialized GOP bundle."""

    # ``GetGOPList`` returns an opaque binary protocol, not numeric image data.
    # Do *not* use ``np.asarray(..., dtype=np.uint8)`` here: for non-uint8
    # buffer exporters that performs a value conversion and can turn every
    # non-zero header byte into ``0x01``.  Copy the underlying bytes verbatim.
    try:
        # This ACCV build exposes its capsule-backed one-dimensional array
        # with an erroneous stride.  ``tobytes()``, ``ascontiguousarray()``,
        # and a memoryview all follow that stride and repeat the first byte
        # (e.g. turning a header of ``01 00 00 00`` into ``01 01 01 01``).
        # The data pointer and nbytes remain correct, so copy the opaque
        # serialized protocol directly from the underlying allocation.
        source = np.asarray(data)
        # ``ctypes.string_at`` would first allocate a Python ``bytes`` object
        # and ``np.frombuffer(...).copy()`` would then allocate the final array.
        # Copy directly into the owned ndarray instead.
        bundle = np.empty(int(source.nbytes), dtype=np.uint8)
        if bundle.nbytes:
            ctypes.memmove(
                int(bundle.ctypes.data),
                int(source.ctypes.data),
                int(bundle.nbytes),
            )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context}: GetGOPList did not return a readable NumPy buffer") from exc
    header_size = struct.calcsize("I")
    offset_size = struct.calcsize("P")
    if bundle.nbytes < header_size:
        raise RuntimeError(f"{context}: serialized GOP is missing its frame-count header")
    frame_count = struct.unpack_from("I", bundle, 0)[0]
    required_size = header_size + frame_count * offset_size
    if frame_count == 0 or bundle.nbytes < required_size:
        raise RuntimeError(
            f"{context}: invalid serialized GOP header "
            f"(frames={frame_count}, bytes={bundle.nbytes}, required>={required_size})"
        )
    return bundle


@dataclass(frozen=True)
class NvcGopRequest:
    """Deferred multi-camera frame request emitted by a DataLoader worker."""

    video_paths: tuple[str, ...]
    frame_ids: tuple[tuple[int, ...], ...]
    camera_features: tuple[str, ...]
    gop_refs: tuple[tuple[Any, ...], ...] | None = None

    def with_gop_refs(self, gop_refs: list[list[Any]]) -> "NvcGopRequest":
        return replace(self, gop_refs=tuple(tuple(refs) for refs in gop_refs))


def make_store_id(global_rank: int = 0) -> int:
    """Return a rank- and process-specific ACCV shared-store identifier."""

    job_id = os.environ.get("SLURM_JOB_ID")
    job_component = int(job_id) if job_id and job_id.isdigit() else os.getpid()
    return job_component * 100_000 + (os.getpid() % 10_000) * 10 + int(global_rank)


def calculate_store_capacity(
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    num_cameras: int,
    num_frames: int,
) -> int:
    """Bound capacity so queued DataLoader requests cannot be evicted early."""

    # PyTorch's ``num_workers * prefetch_factor`` already includes batches
    # being materialized by each worker.  The main process needs one additional
    # batch while it resolves references and submits NVDEC; once submitted, the
    # decoder owns the GOP bytes and queued materialized batches no longer hold
    # SharedGopStore references.
    queued_batches = max(1, num_workers * prefetch_factor) + 1
    return max(1, queued_batches * batch_size * num_cameras * num_frames)


def materialize_gop_request(
    request: NvcGopRequest,
    *,
    store: Any,
    demuxer: Any,
) -> NvcGopRequest:
    """Demux missing GOPs in a worker and store their bytes in shared memory.

    ``store`` is an attached ``accvlab.on_demand_video_decoder.SharedGopStore``;
    ``demuxer`` is a worker-local ``CreateGopDecoder``.  Both are injected so
    worker lifecycle remains controlled by the recipe dataset.
    """

    if request.gop_refs is not None:
        return request
    refs_by_video: list[list[Any]] = []
    for video_path, frame_ids in zip(request.video_paths, request.frame_ids):
        unique_refs: dict[tuple[int, int, str], Any] = {}
        for frame_id in frame_ids:
            ref = store.lookup(video_path, int(frame_id))
            if ref is None:
                # The ACCV GOP cache is local to this worker.  Its returned
                # view is copied into an owned bundle below before it enters
                # SharedGopStore, so no process-local view crosses a process
                # boundary.
                numpy_data, first_ids, gop_lengths = demuxer.GetGOPList(
                    [video_path], [int(frame_id)], useGOPCache=True
                )[0]
                if len(first_ids) != 1 or len(gop_lengths) != 1:
                    raise RuntimeError(f"Unexpected GOP metadata for {video_path}")
                bundle = own_serialized_gop_bundle(
                    numpy_data,
                    context=f"GetGOPList({video_path}, frame_id={frame_id})",
                )
                ref = store.put(video_path, int(first_ids[0]), int(gop_lengths[0]), bundle)
            unique_refs[(int(ref.first_frame_id), int(ref.gop_len), ref.shm_name)] = ref
        refs_by_video.append([unique_refs[key] for key in sorted(unique_refs)])
    return request.with_gop_refs(refs_by_video)
