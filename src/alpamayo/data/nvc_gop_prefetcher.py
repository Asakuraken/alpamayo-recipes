"""Deferred ACCV-Lab GOP loading and main-process asynchronous decode.

The data loader workers only demux compressed GOP byte ranges and put them in
``SharedGopStore``.  This module resolves the resulting lightweight references
in the training process, submits a whole batch to NVDEC, and materializes the
VLM image inputs on the GPU.

The iterator owns a bounded queue of materialized batches. Decoded frames are
cloned out of ACCV-Lab's reusable output pool before the next decode is
submitted, while the producer continues filling the next gradient-accumulation
group during model work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
import queue
import threading
import time
from typing import Any, Iterator

import torch

from alpamayo.data.nvc_gop import (
    NVC_GOP_REQUEST_KEY,
    NvcGopRequest,
    calculate_store_capacity,
    make_store_id,
)


@dataclass
class _VideoOccurrence:
    group_index: int
    frame_positions: tuple[int, ...]


@dataclass
class _PendingDecode:
    batch: Any
    requests: list[NvcGopRequest]
    video_paths: list[str]
    frame_ids: list[list[int]]
    frames_per_video: list[int]
    sample_occurrences: list[list[_VideoOccurrence]]


@dataclass(frozen=True)
class _WorkerFailure:
    exception: BaseException


_END_OF_DATA = object()


def _record_cuda_tensors(
    value: Any,
    stream: torch.cuda.Stream,
    seen: set[int] | None = None,
) -> None:
    """Keep nested CUDA tensor storage alive until ``stream`` is done with it."""

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return

    if isinstance(value, torch.Tensor):
        seen.add(value_id)
        if value.is_cuda:
            value.record_stream(stream)
        return
    if isinstance(value, Mapping):
        seen.add(value_id)
        for item in value.values():
            _record_cuda_tensors(item, stream, seen)
        return
    if isinstance(value, (list, tuple)):
        seen.add(value_id)
        for item in value:
            _record_cuda_tensors(item, stream, seen)

def _unlink_store_orphans(store_id: int) -> int:
    """Remove evicted GOP blocks left behind by ACCV-Lab's native store.

    ``SharedGopStore.cleanup()`` releases the entries that remain indexed, but
    older native builds do not unlink blocks that were evicted from the index.
    All workers are stopped before this helper runs, so files under the exact
    per-rank store prefix are no longer in use.
    """

    prefix = f"gs_{int(store_id)}_"
    removed = 0
    try:
        entries = os.scandir("/dev/shm")
    except FileNotFoundError:
        return 0
    with entries:
        for entry in entries:
            if not entry.name.startswith(prefix):
                continue
            try:
                os.unlink(entry.path)
                removed += 1
            except FileNotFoundError:
                pass
    return removed


def _shutdown_dataloader_workers(dataloader: Any) -> bool:
    """Terminate workers through an Accelerate wrapper, if one is present."""

    seen: set[int] = set()
    candidate = dataloader
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        shutdown_workers = getattr(candidate, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
            return True
        raw_iterator = getattr(candidate, "_iterator", None)
        shutdown_workers = getattr(raw_iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
            return True
        candidate = getattr(candidate, "base_dataloader", None)
    return False


def _persistent_workers_enabled(dataloader: Any) -> bool:
    """Read ``persistent_workers`` through an optional Accelerate wrapper."""

    seen: set[int] = set()
    candidate = dataloader
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        persistent_workers = getattr(candidate, "persistent_workers", None)
        if persistent_workers is not None:
            return bool(persistent_workers)
        candidate = getattr(candidate, "base_dataloader", None)
    return False


class NvcGopBatchPrefetcher:
    """Wrap a DataLoader with ACCV-Lab GOP resolution and bounded prefetching."""

    def __init__(
        self,
        dataloader,
        *,
        dataset,
        processor,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int = 2,
        prefetch_batches: int = 1,
        store_capacity: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        try:
            import accvlab.on_demand_video_decoder as nvc
        except ImportError as exc:  # pragma: no cover - exercised in ACCV environment
            raise ImportError(
                "The nvc backend requires accvlab.on_demand_video_decoder."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("The nvc GOP pipeline requires a CUDA device")

        self.dataloader = dataloader
        self.dataset = dataset
        self.processor = processor
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.prefetch_batches = int(prefetch_batches)
        if self.prefetch_batches < 1:
            raise ValueError("prefetch_batches must be positive")
        self.device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
        self.gpu_id = self.device.index if self.device.index is not None else 0
        self._nvc = nvc

        max_videos, max_frames = dataset.get_nvc_gop_shape()
        self.max_videos_per_sample = max_videos
        self.max_frames_per_video = max_frames
        # A small group amortizes repeated GOP decode while retaining enough
        # independent decoder rows for NVDEC concurrency. Grouping an entire
        # per-rank batch into four rows was measurably slower because it
        # serialized too much decode work.
        self.max_grouped_frames = max(2, self.max_frames_per_video)
        self.store_capacity = int(
            store_capacity
            or calculate_store_capacity(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                num_cameras=max_videos,
                num_frames=max_frames,
            )
        )
        if self.store_capacity < 1:
            raise ValueError("nvc_gop_store_capacity must be positive")

        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.store_id = make_store_id(global_rank)
        self.dataset.configure_nvc_gop_store(
            store_id=self.store_id,
            capacity=self.store_capacity,
        )
        # This must happen before iter(dataloader), which starts spawned workers.
        self._store = nvc.SharedGopStore.create(
            capacity=self.store_capacity,
            store_id=self.store_id,
        )
        self._decoder = nvc.CreateBatchAsyncGopDecoder(
            maxfiles=self.batch_size * self.max_videos_per_sample,
            # Adjacent samples from a shard usually reference the same camera
            # files. Group those occurrences into one decoder row and request
            # a small frame chunk from that file in one call.
            max_frames_per_decode_call=self.max_grouped_frames,
            iGpu=self.gpu_id,
        )
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._transform_stream = torch.cuda.Stream(device=self.device)
        self._closed = False
        self._active_iterator = False
        self._iterator = None

        logging.info(
            "Initialized nvc GOP pipeline: store_id=%s capacity=%s maxfiles=%s "
            "max_frames=%s grouped_frames=%s prefetch_batches=%s device=%s",
            self.store_id,
            self.store_capacity,
            self.batch_size * self.max_videos_per_sample,
            self.max_frames_per_video,
            self.max_grouped_frames,
            self.prefetch_batches,
            self.device,
        )

    def __getattr__(self, name: str):
        # Preserve DataLoader attributes used by Trainer/Accelerate.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.dataloader, name)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self) -> Iterator[Any]:
        if self._closed:
            raise RuntimeError("NvcGopBatchPrefetcher has already been closed")
        if self._active_iterator:
            # Trainer can end an epoch exactly after consuming its final
            # training batch, without asking the iterable for one more item
            # (and therefore without driving ``__next__`` to StopIteration).
            # Close that stale epoch iterator before constructing the next
            # one.  This is sequential epoch rollover, not concurrent use.
            if self._iterator is None:
                raise RuntimeError("NVC GOP iterator state is inconsistent")
            self._iterator.close()
            self._iterator = None
        self._active_iterator = True
        self._iterator = _NvcGopIterator(self)
        return self._iterator

    def _submit(self, batch: Any) -> _PendingDecode:
        return self._submit_impl(batch)

    def _submit_impl(self, batch: Any) -> _PendingDecode:
        inputs = batch["inputs"]
        requests = inputs.get(NVC_GOP_REQUEST_KEY)
        if not requests:
            raise RuntimeError(f"Raw nvc batch is missing {NVC_GOP_REQUEST_KEY}")
        if not all(isinstance(request, NvcGopRequest) for request in requests):
            raise TypeError("Invalid nvc GOP request in collated batch")
        if not all(request.gop_refs is not None for request in requests):
            raise RuntimeError("DataLoader worker returned unresolved GOP references")

        # Group repeated camera files across samples. Sharded batches commonly
        # contain adjacent steps from one episode, so small chunks reuse GOP
        # decode work without collapsing the batch to too few NVDEC streams.
        path_to_group: dict[str, int] = {}
        video_paths: list[str] = []
        frame_ids: list[list[int]] = []
        refs_by_group: list[dict[tuple[int, int, str], Any]] = []
        sample_occurrences: list[list[_VideoOccurrence]] = []
        for request in requests:
            assert request.gop_refs is not None
            if not (
                len(request.video_paths)
                == len(request.frame_ids)
                == len(request.gop_refs)
                == len(request.camera_features)
            ):
                raise RuntimeError("Inconsistent camera dimensions in nvc GOP request")
            occurrences: list[_VideoOccurrence] = []
            for path, ids, refs in zip(
                request.video_paths, request.frame_ids, request.gop_refs
            ):
                if not ids:
                    raise RuntimeError("nvc GOP request contains an empty frame list")
                group_index = path_to_group.get(path)
                if (
                    group_index is None
                    or len(frame_ids[group_index]) + len(ids) > self.max_grouped_frames
                ):
                    group_index = len(video_paths)
                    path_to_group[path] = group_index
                    video_paths.append(path)
                    frame_ids.append([])
                    refs_by_group.append({})
                positions = tuple(
                    range(
                        len(frame_ids[group_index]),
                        len(frame_ids[group_index]) + len(ids),
                    )
                )
                frame_ids[group_index].extend(int(frame_id) for frame_id in ids)
                for ref in refs:
                    refs_by_group[group_index][
                        (int(ref.first_frame_id), int(ref.gop_len), ref.shm_name)
                    ] = ref
                occurrences.append(
                    _VideoOccurrence(
                        group_index=group_index,
                        frame_positions=positions,
                    )
                )
            sample_occurrences.append(occurrences)

        frames_per_video = [len(ids) for ids in frame_ids]
        max_frames = max(frames_per_video)
        decoder_frame_limit = self.max_grouped_frames
        if max_frames > decoder_frame_limit:
            raise RuntimeError(
                f"Grouped batch requests {max_frames} frames/video, decoder limit is "
                f"{decoder_frame_limit}"
            )
        # ACCV-Lab requires rectangular [video][frame] input. Repeat the last
        # ID for shorter groups and ignore those padding positions on collect.
        padded_frame_ids = [
            ids + [ids[-1]] * (max_frames - len(ids)) for ids in frame_ids
        ]

        flat_refs: list[Any] = []
        ref_counts: list[int] = []
        for refs in refs_by_group:
            ordered_refs = [refs[key] for key in sorted(refs)]
            flat_refs.extend(ordered_refs)
            ref_counts.append(len(ordered_refs))

        shared_arrays = self._store.get_batch(flat_refs)
        if len(shared_arrays) != len(flat_refs):
            raise RuntimeError(
                f"SharedGopStore returned {len(shared_arrays)} bundles for {len(flat_refs)} refs"
            )
        # DecodeFromGOPListRGB synchronously copies each serialized bundle at
        # submission.  Keep the SharedGopStore views alive only for that call;
        # copying them into Python-owned arrays here adds a full CPU-memory pass
        # for every unique GOP without extending their required lifetime.
        for ref, shared in zip(flat_refs, shared_arrays):
            if int(shared.nbytes) != int(ref.data_size):
                raise RuntimeError(
                    f"SharedGopStore size mismatch for {ref.shm_name}: "
                    f"ref={ref.data_size}, read={shared.nbytes}"
                )
        numpy_datas: list[list[Any]] = []
        offset = 0
        for count in ref_counts:
            numpy_datas.append(shared_arrays[offset : offset + count])
            offset += count

        self._decoder.DecodeFromGOPListRGB(
            numpy_datas,
            video_paths,
            padded_frame_ids,
            False,
        )
        return _PendingDecode(
            batch=batch,
            requests=requests,
            video_paths=video_paths,
            frame_ids=padded_frame_ids,
            frames_per_video=frames_per_video,
            sample_occurrences=sample_occurrences,
        )

    def _collect_and_clone(self, pending: _PendingDecode) -> list[torch.Tensor]:
        decoded = self._decoder.DecodeFromGOPListRGBGetBuffer(
            pending.video_paths,
            pending.frame_ids,
            False,
        )
        if len(decoded) != len(pending.video_paths):
            raise RuntimeError(
                f"ACCV-Lab returned {len(decoded)} video rows for "
                f"{len(pending.video_paths)} requests"
            )

        # ``decoded`` points into ACCV-Lab's reusable output pool. Build the
        # final owned [T*V, C, H, W] sample tensors directly from those views.
        # The previous implementation first cloned every HWC frame and then
        # stacked the CHW views, copying the full decoded batch twice.
        samples: list[torch.Tensor] = []
        with torch.cuda.stream(self._copy_stream):
            for occurrences in pending.sample_occurrences:
                if not occurrences:
                    raise RuntimeError("nvc GOP request contains no camera views")
                temporal_length = len(occurrences[0].frame_positions)
                if not all(
                    len(occurrence.frame_positions) == temporal_length
                    for occurrence in occurrences
                ):
                    raise RuntimeError(
                        "Camera views in one sample have different temporal lengths"
                    )
                for occurrence in occurrences:
                    row = decoded[occurrence.group_index]
                    if not occurrence.frame_positions:
                        raise RuntimeError("nvc GOP request contains no frame positions")
                    last_position = occurrence.frame_positions[-1]
                    if len(row) <= last_position:
                        raise RuntimeError(
                            f"ACCV-Lab returned {len(row)} frames for video "
                            f"{occurrence.group_index}; expected position "
                            f"{last_position}"
                        )
                # Match the legacy processor order: [T, V, C, H, W] ->
                # [T*V, C, H, W]. torch.stack owns the result, so no separate
                # clone of each decoder-backed frame is necessary.
                sample_frames = [
                    torch.as_tensor(
                        decoded[occurrence.group_index][
                            occurrence.frame_positions[time]
                        ],
                        device=self.device,
                    ).permute(2, 0, 1)
                    for time in range(temporal_length)
                    for occurrence in occurrences
                ]
                samples.append(torch.stack(sample_frames, dim=0))

        # ACCV-Lab owns and reuses ``decoded``. Its next submit runs on an
        # internal stream that cannot wait on a PyTorch event, so only the
        # small D2D ownership copy is synchronized here. Image transforms stay
        # asynchronous and overlap the next NVDEC submission.
        self._copy_stream.synchronize()

        return samples

    def _materialize(
        self, pending: _PendingDecode, samples: list[torch.Tensor]
    ) -> tuple[Any, torch.cuda.Event]:
        materialize_batch = getattr(self.processor, "materialize_batch", None)
        if materialize_batch is not None:
            with torch.cuda.stream(self._transform_stream):
                _record_cuda_tensors(samples, self._transform_stream)
                materialized = materialize_batch(pending.batch, samples)
                ready_event = torch.cuda.Event()
                ready_event.record(self._transform_stream)
            return materialized, ready_event
        inputs = pending.batch["inputs"]
        inputs.pop(NVC_GOP_REQUEST_KEY)
        with torch.cuda.stream(self._transform_stream):
            _record_cuda_tensors(samples, self._transform_stream)
            vlm_inputs = self.processor.collate_decoded_vlm_inputs(
                decoded_samples=samples,
                languages=["" for _ in pending.requests],
                device=self.device,
            )
            ready_event = torch.cuda.Event()
            ready_event.record(self._transform_stream)
        inputs.update(vlm_inputs)
        return pending.batch, ready_event

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_start = time.perf_counter()
        raw_workers_shutdown = False
        if self._iterator is not None:
            try:
                # Epoch teardown leaves persistent DataLoader workers alive,
                # but final wrapper teardown must also release their store
                # attachments before the store is cleaned.
                self._iterator.close(shutdown_workers=True)
                raw_workers_shutdown = True
            except BaseException:
                logging.exception(
                    "Failed while draining nvc background prefetch during cleanup"
                )
            self._iterator = None
        # A completed epoch can have cleared our wrapper reference while a
        # persistent PyTorch DataLoader still retains its raw iterator. Final
        # teardown must release those workers before the shared store is removed.
        if not raw_workers_shutdown:
            _shutdown_dataloader_workers(self.dataloader)
        iterator_closed = time.perf_counter()
        try:
            if self._decoder is not None:
                release_memory = getattr(self._decoder, "release_device_memory", None)
                if release_memory is not None:
                    release_memory()
                release_decoder = getattr(self._decoder, "release_decoder", None)
                if release_decoder is not None:
                    release_decoder()
        finally:
            self._decoder = None
            decoder_closed = time.perf_counter()
            if self._store is not None:
                self._store.cleanup()
                self._store = None
            orphan_count = _unlink_store_orphans(self.store_id)
            store_closed = time.perf_counter()
            self.dataset.clear_nvc_gop_store_configuration()
            cleanup_done = time.perf_counter()
            logging.info(
                "Closed nvc GOP pipeline in %.2fs "
                "(iterator=%.2fs decoder=%.2fs store=%.2fs dataset=%.2fs, "
                "unlinked_orphans=%s)",
                cleanup_done - close_start,
                iterator_closed - close_start,
                decoder_closed - iterator_closed,
                store_closed - decoder_closed,
                cleanup_done - store_closed,
                orphan_count,
            )


class _NvcGopIterator:
    def __init__(self, owner: NvcGopBatchPrefetcher) -> None:
        self.owner = owner
        self._closed = False
        self.raw_iterator = None
        self.worker = None
        self._raw_workers_shutdown = False
        self._raw_iterator_closed = False
        try:
            self.raw_iterator = iter(owner.dataloader)
            self.worker = _DecodeTransformWorker(
                owner,
                self.raw_iterator,
                capacity=owner.prefetch_batches,
            )
            self.worker.start()
        except BaseException:
            if self.worker is not None:
                self.worker.request_stop()
            self._shutdown_raw_workers()
            if self.worker is not None:
                self.worker.join()
            self._close_raw_iterator()
            self._closed = True
            self.owner._active_iterator = False
            raise

    def __iter__(self):
        return self

    def __next__(self):
        result = self.worker.wait()
        if result is None:
            self.close()
            raise StopIteration
        batch, ready_event = result
        consumer_stream = torch.cuda.current_stream(self.owner.device)
        consumer_stream.wait_event(ready_event)
        _record_cuda_tensors(batch, consumer_stream)
        return batch

    def __del__(self):
        self.close()

    def _shutdown_raw_workers(self) -> None:
        """Permanently terminate the underlying DataLoader workers once."""

        if self._raw_workers_shutdown:
            return
        self._raw_workers_shutdown = True
        shutdown_workers = getattr(self.raw_iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
            return
        _shutdown_dataloader_workers(self.owner.dataloader)

    def _close_raw_iterator(self) -> None:
        """Close an Accelerate generator after its consumer thread has stopped."""

        if getattr(self, "_raw_iterator_closed", True):
            return
        self._raw_iterator_closed = True
        close = getattr(self.raw_iterator, "close", None)
        if close is not None:
            close()

    def close(self, *, shutdown_workers: bool | None = None) -> None:
        """Finish one epoch without defeating persistent DataLoader workers."""

        if self._closed:
            if shutdown_workers:
                self._shutdown_raw_workers()
            self._close_raw_iterator()
            return
        if shutdown_workers is None:
            shutdown_workers = not _persistent_workers_enabled(self.owner.dataloader)
        self._closed = True
        try:
            if self.worker is not None:
                self.worker.request_stop()
            # Worker shutdown unblocks a background thread waiting in
            # ``next(raw_iterator)``. It is intentionally deferred for
            # persistent workers until NvcGopBatchPrefetcher.close().
            if shutdown_workers:
                self._shutdown_raw_workers()
            if self.worker is not None:
                self.worker.join()
            self._close_raw_iterator()
        finally:
            self.owner._active_iterator = False


class _DecodeTransformWorker:
    """Continuously fetch, decode, transform, and enqueue materialized batches."""

    def __init__(
        self,
        owner: NvcGopBatchPrefetcher,
        raw_iterator: Iterator[Any],
        *,
        capacity: int,
    ) -> None:
        self.owner = owner
        self.raw_iterator = raw_iterator
        self.results: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self._stop_requested = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("NVC background prefetch worker has already started")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _publish(self, result: object) -> bool:
        while not self._stop_requested.is_set():
            try:
                self.results.put(result, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self.owner.device)
            while not self._stop_requested.is_set():
                raw_batch = next(self.raw_iterator)
                pending = self.owner._submit(raw_batch)
                owned_samples = self.owner._collect_and_clone(pending)
                result = self.owner._materialize(pending, owned_samples)
                if not self._publish(result):
                    return
        except StopIteration:
            self._publish(_END_OF_DATA)
        except BaseException as exc:
            self._publish(_WorkerFailure(exc))

    def wait(self) -> tuple[Any, torch.cuda.Event] | None:
        if self.thread is None:
            return None
        result = self.results.get()
        if result is _END_OF_DATA:
            return None
        if isinstance(result, _WorkerFailure):
            raise result.exception
        return result

    def request_stop(self) -> None:
        self._stop_requested.set()

    def join(self) -> None:
        if self.thread is None:
            return
        self.thread.join()
        self.thread = None
        # Drop any queued GPU tensors promptly during iterator cleanup.
        while True:
            try:
                self.results.get_nowait()
            except queue.Empty:
                break
