"""Manifest-backed PAI dataset for ACCV-Lab's deferred GOP decoder."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch

from alpamayo.data.nvc_gop import NVC_GOP_REQUEST_KEY, NvcGopRequest, materialize_gop_request
from alpamayo.data.pai import PAIDataset


def _load_trajectory_without_images(
    *,
    clip_id: str,
    t0_us: int,
    avdi: Any,
    num_history_steps: int,
    num_future_steps: int,
    time_step: float,
) -> dict[str, Any]:
    """Load PAI trajectory targets without invoking the JPEG image loader.

    The released ``load_physical_aiavdataset`` API always decodes camera
    images and the version pinned by this recipe has no ``load_images`` flag.
    NVC owns image loading, so duplicate CPU decoding here would defeat the
    demuxer/decoder separation pipeline.
    """

    egomotion = avdi.get_clip_feature(
        clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True
    )
    history_time_range_us = num_history_steps * time_step * 1_000_000
    if t0_us <= history_time_range_us:
        raise ValueError(
            f"{t0_us=} must be greater than the history time range "
            f"({history_time_range_us=} us)"
        )
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    history = egomotion(t0_us + history_offsets_us)
    future = egomotion(t0_us + future_offsets_us)
    t0_xyz = history.pose.translation[-1].copy()
    t0_rotation = spt.Rotation.from_quat(history.pose.rotation.as_quat()[-1]).inv()

    def local_pose(pose: Any) -> tuple[torch.Tensor, torch.Tensor]:
        xyz = t0_rotation.apply(pose.translation - t0_xyz)
        rotation = (t0_rotation * spt.Rotation.from_quat(pose.rotation.as_quat())).as_matrix()
        return (
            torch.from_numpy(xyz).float().unsqueeze(0).unsqueeze(0),
            torch.from_numpy(rotation).float().unsqueeze(0).unsqueeze(0),
        )

    history_xyz, history_rot = local_pose(history.pose)
    future_xyz, future_rot = local_pose(future.pose)
    return {
        "ego_history_xyz": history_xyz,
        "ego_history_rot": history_rot,
        "ego_future_xyz": future_xyz,
        "ego_future_rot": future_rot,
        "t0_us": t0_us,
        "clip_id": clip_id,
    }


CAMERA_INDICES = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_front_tele_30fov": 6,
}
CAMERA_FEATURES = tuple(CAMERA_INDICES)


class NvcPAIDataset(PAIDataset):
    """PAI samples whose image tensors are materialized in the main process.

    The default :class:`PAIDataset` remains unchanged.  This class returns all
    non-visual fields plus a lightweight request; DataLoader workers replace
    that request with SharedGopStore references immediately before collation.
    """

    def __init__(self, *, nvc_manifest: str, num_frames: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_frames = int(num_frames)
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")
        self._manifest = self._read_manifest(nvc_manifest)
        self._timestamp_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._nvc_store = None
        self._nvc_demuxer = None
        self._nvc_store_id: int | None = None
        self._nvc_store_capacity: int | None = None

    @staticmethod
    def _read_manifest(path: str) -> dict[tuple[str, str], dict[str, Any]]:
        records: dict[tuple[str, str], dict[str, Any]] = {}
        with Path(path).open() as stream:
            for line in stream:
                record = json.loads(line)
                records[(record["clip_id"], record["camera_feature"])] = record
        if not records:
            raise ValueError(f"NVC manifest is empty: {path}")
        return records

    def configure_nvc_gop_store(self, store_id: int, capacity: int) -> None:
        self.close_nvc_gop_worker_resources()
        self._nvc_store_id, self._nvc_store_capacity = int(store_id), int(capacity)

    def clear_nvc_gop_store_configuration(self) -> None:
        self.close_nvc_gop_worker_resources()
        self._nvc_store_id = self._nvc_store_capacity = None

    def close_nvc_gop_worker_resources(self) -> None:
        if self._nvc_store is not None:
            self._nvc_store.close()
            self._nvc_store = None
        if self._nvc_demuxer is not None:
            for method in ("release_device_memory", "release_decoder"):
                release = getattr(self._nvc_demuxer, method, None)
                if release is not None:
                    release()
            self._nvc_demuxer = None

    def get_nvc_gop_shape(self) -> tuple[int, int]:
        return len(CAMERA_FEATURES), self.num_frames

    def _timestamps_for_record(self, record: dict[str, Any]) -> np.ndarray:
        """Load and cache a camera timestamp column from a manifest record.

        Extracted-video manifests point ``timestamps_path`` at a standalone
        parquet file.  ZIP-subfile manifests instead retain the original PAI
        archive and specify ``timestamps_zip_path`` plus
        ``timestamps_member``.  The latter is intentionally read into a
        short-lived :class:`io.BytesIO`: timestamp parquets are small, while
        retaining an open ``ZipFile`` in a DataLoader worker complicates
        pickling and worker teardown.
        """

        timestamp_path = record.get("timestamps_path")
        if timestamp_path:
            cache_key = ("path", timestamp_path)
            timestamps = self._timestamp_cache.get(cache_key)
            if timestamps is None:
                timestamps = pd.read_parquet(timestamp_path)["timestamp"].to_numpy(
                    dtype=np.int64
                )
                self._timestamp_cache[cache_key] = timestamps
            return timestamps

        timestamps_zip_path = record.get("timestamps_zip_path")
        timestamps_member = record.get("timestamps_member")
        if not timestamps_zip_path or not timestamps_member:
            raise KeyError(
                "NVC manifest record requires either timestamps_path or both "
                "timestamps_zip_path and timestamps_member"
            )

        cache_key = ("zip", timestamps_zip_path, timestamps_member)
        timestamps = self._timestamp_cache.get(cache_key)
        if timestamps is None:
            with zipfile.ZipFile(timestamps_zip_path) as archive:
                try:
                    timestamp_bytes = archive.read(timestamps_member)
                except KeyError as exc:
                    raise KeyError(
                        f"Timestamp member {timestamps_member!r} was not found in "
                        f"{timestamps_zip_path!r}"
                    ) from exc
            timestamps = pd.read_parquet(io.BytesIO(timestamp_bytes))["timestamp"].to_numpy(
                dtype=np.int64
            )
            self._timestamp_cache[cache_key] = timestamps
        return timestamps

    def _frame_ids(
        self, clip_id: str, t0_us: int
    ) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], torch.Tensor, torch.Tensor]:
        requested = np.asarray(
            [
                t0_us - (self.num_frames - 1 - i) * int(self.time_step * 1_000_000)
                for i in range(self.num_frames)
            ],
            dtype=np.int64,
        )
        paths, ids, actual_timestamps = [], [], []
        for feature in CAMERA_FEATURES:
            record = self._manifest.get((clip_id, feature))
            if record is None:
                raise KeyError(f"NVC manifest has no {feature} video for clip {clip_id}")
            timestamps = self._timestamps_for_record(record)
            if timestamps.ndim != 1 or not len(timestamps):
                raise ValueError(
                    f"Timestamp data for {feature} video in clip {clip_id} must be non-empty and 1D"
                )
            if np.any(timestamps[1:] <= timestamps[:-1]):
                raise ValueError(
                    f"Timestamp data for {feature} video in clip {clip_id} must be "
                    "strictly increasing"
                )
            if requested.min() < timestamps[0] or requested.max() > timestamps[-1]:
                raise ValueError(
                    "Requested timestamps must be within the range of timestamps for "
                    f"{feature} video in clip {clip_id}:\n"
                    f"{requested.min()=}, {requested.max()=}\n"
                    f"{timestamps[0]=}, {timestamps[-1]=}"
                )
            # Match physical_ai_av.video.VideoReader: use the closest frame at
            # or before each request, rather than the absolute nearest frame.
            chosen = np.searchsorted(timestamps, requested, side="right") - 1
            paths.append(record["video_path"])
            ids.append(tuple(int(item) for item in chosen))
            actual_timestamps.append(torch.from_numpy(timestamps[chosen]))
        absolute = torch.stack(actual_timestamps)
        relative = (absolute - absolute.min()).float() * 1e-6
        camera_indices = torch.tensor(
            [CAMERA_INDICES[key] for key in CAMERA_FEATURES], dtype=torch.int64
        )
        return tuple(paths), tuple(ids), camera_indices, relative

    def _get_sample(self, clip_id: str, t0_us: int) -> dict[str, Any]:
        sample = _load_trajectory_without_images(
            clip_id=clip_id,
            t0_us=t0_us,
            avdi=self.avdi,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
        )
        for key in tuple(sample):
            if key.startswith("ego_"):
                sample[key] = sample[key].squeeze(0)
        paths, frame_ids, camera_indices, relative = self._frame_ids(clip_id, int(t0_us))
        sample["camera_indices"] = camera_indices
        sample["relative_timestamps"] = relative
        sample[NVC_GOP_REQUEST_KEY] = NvcGopRequest(
            video_paths=paths, frame_ids=frame_ids, camera_features=CAMERA_FEATURES
        )
        if self.avdi.reasoning_db is not None:
            sample.update(self.avdi.get_reasoning_data(clip_id, t0_us))
        return self.materialize_nvc_sample(sample) if self._nvc_store_id is not None else sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_id = self.clip_ids[idx]
        t0_us = (
            self.DEFAULT_T0_US
            if self.use_default_keyframe
            else self.avdi.get_clip_key_frame(clip_id)
        )
        return self._get_sample(clip_id, int(t0_us))

    def materialize_nvc_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self._nvc_store_id is None or self._nvc_store_capacity is None:
            raise RuntimeError("NVC SharedGopStore was not configured before worker startup")
        try:
            import accvlab.on_demand_video_decoder as nvc
        except ImportError as exc:
            raise ImportError("Install accvlab.on_demand_video_decoder for nvc_gop") from exc
        if self._nvc_store is None or self._nvc_demuxer is None:
            if self._nvc_store is None:
                self._nvc_store = nvc.SharedGopStore.attach(
                    self._nvc_store_capacity, self._nvc_store_id
                )
            if self._nvc_demuxer is None:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                self._nvc_demuxer = nvc.CreateGopDecoder(
                    maxfiles=len(CAMERA_FEATURES), iGpu=local_rank
                )
        result = dict(sample)
        result[NVC_GOP_REQUEST_KEY] = materialize_gop_request(
            sample[NVC_GOP_REQUEST_KEY], store=self._nvc_store, demuxer=self._nvc_demuxer
        )
        return result
