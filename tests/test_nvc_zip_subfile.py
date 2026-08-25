"""Regression coverage for NVC manifests that retain PAI camera ZIPs.

These tests deliberately avoid ACCV and CUDA.  They exercise the ZIP layout
and the lightweight manifest/dataset boundary that must agree before an ACCV
decoder ever receives a ``subfile`` URL.
"""

from __future__ import annotations

import bisect
from contextlib import contextmanager
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = ROOT / "scripts" / "prepare_pai_nvc_videos.py"
NVC_PAI_PATH = ROOT / "src" / "alpamayo" / "data" / "nvc_pai.py"


@contextmanager
def _temporary_modules(modules: dict[str, types.ModuleType]):
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, original in previous.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _load_module(path: Path, name: str, stubs: dict[str, types.ModuleType] | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _temporary_modules({**(stubs or {}), name: module}):
        spec.loader.exec_module(module)
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


class _Array:
    """Tiny ndarray stand-in sufficient for the CPU-only frame-id test."""

    ndim = 1

    def __init__(self, values) -> None:
        self.values = list(values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _Array(self.values[item])
        if isinstance(item, _Array):
            return _Array(self.values[int(index)] for index in item.values)
        if isinstance(item, (list, tuple)):
            return _Array(self.values[int(index)] for index in item)
        return self.values[item]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __le__(self, other):
        values = other.values if isinstance(other, _Array) else [other] * len(self)
        return _Array(left <= right for left, right in zip(self.values, values))

    def __sub__(self, other):
        if isinstance(other, _Array):
            return _Array(left - right for left, right in zip(self.values, other.values))
        return _Array(value - other for value in self.values)

    def min(self):
        return min(self.values)

    def max(self):
        return max(self.values)


class _TimestampColumn:
    def __init__(self, values: _Array) -> None:
        self.values = values

    def to_numpy(self, *, dtype):
        del dtype
        return self.values


class _TimestampFrame:
    def __init__(self, values: _Array) -> None:
        self.values = values

    def __getitem__(self, name: str) -> _TimestampColumn:
        assert name == "timestamp"
        return _TimestampColumn(self.values)


def _load_nvc_pai_module(read_parquet):
    numpy = types.ModuleType("numpy")
    numpy.int64 = int

    pandas = types.ModuleType("pandas")
    pandas.read_parquet = read_parquet

    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.int64 = object()

    scipy = _package("scipy")
    scipy_spatial = _package("scipy.spatial")
    scipy_transform = types.ModuleType("scipy.spatial.transform")
    scipy.spatial = scipy_spatial
    scipy_spatial.transform = scipy_transform

    nvc_gop = types.ModuleType("alpamayo.data.nvc_gop")
    nvc_gop.NVC_GOP_REQUEST_KEY = "_nvc_gop_request"
    nvc_gop.NvcGopRequest = object
    nvc_gop.materialize_gop_request = lambda request, **_kwargs: request

    pai = types.ModuleType("alpamayo.data.pai")

    class _PAIDataset:
        pass

    pai.PAIDataset = _PAIDataset

    return _load_module(
        NVC_PAI_PATH,
        "_nvc_pai_zip_subfile_test",
        {
            "numpy": numpy,
            "pandas": pandas,
            "torch": torch,
            "scipy": scipy,
            "scipy.spatial": scipy_spatial,
            "scipy.spatial.transform": scipy_transform,
            "alpamayo": _package("alpamayo"),
            "alpamayo.data": _package("alpamayo.data"),
            "alpamayo.data.nvc_gop": nvc_gop,
            "alpamayo.data.pai": pai,
        },
    )


def _write_features_csv(root: Path, feature: str) -> None:
    (root / "features.csv").write_text(
        "feature,chunk_path\n"
        f"{feature},{feature}/{feature}.chunk_{{chunk_id:04d}}.zip\n",
        encoding="utf-8",
    )


def _write_camera_archive(
    root: Path,
    *,
    feature: str,
    clip_id: str,
    video_compression: int = zipfile.ZIP_STORED,
    timestamp_compression: int = zipfile.ZIP_STORED,
) -> tuple[Path, str, str, bytes, bytes]:
    archive = root / feature / f"{feature}.chunk_0001.zip"
    archive.parent.mkdir(parents=True)
    video_member = f"{clip_id}.{feature}.mp4"
    timestamps_member = f"{clip_id}.{feature}.timestamps.parquet"
    video_bytes = b"not-a-real-mp4-but-an-exact-member-payload" * 3
    timestamps_bytes = b"timestamp-parquet-payload" * 2
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(video_member, video_bytes, compress_type=video_compression)
        handle.writestr(
            timestamps_member,
            timestamps_bytes,
            compress_type=timestamp_compression,
        )
    return archive, video_member, timestamps_member, video_bytes, timestamps_bytes


def _member_payload_offset(archive: Path, info: zipfile.ZipInfo) -> int:
    """Read the local ZIP header so this stays correct with extra fields."""

    with archive.open("rb") as stream:
        stream.seek(info.header_offset)
        local_header = stream.read(30)
    assert len(local_header) == 30
    assert struct.unpack_from("<I", local_header)[0] == 0x04034B50
    filename_length, extra_length = struct.unpack_from("<HH", local_header, 26)
    return info.header_offset + 30 + filename_length + extra_length


class ZipSubfileManifestTests(unittest.TestCase):
    FEATURE = "camera_front_wide_120fov"
    CLIP_ID = "clip-001"

    def _run_prepare(self, dataset_root: Path, output_root: Path) -> None:
        prepare = _load_module(PREPARE_PATH, "_prepare_nvc_zip_subfile_test")
        argv = [
            "prepare_pai_nvc_videos.py",
            str(dataset_root),
            str(output_root),
            "--mode",
            "zip-subfile",
        ]
        with mock.patch.object(sys, "argv", argv), io.StringIO() as output:
            with mock.patch("sys.stdout", output):
                prepare.main()

    def test_zip_subfile_manifest_points_to_stored_member_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary) / "dataset"
            output_root = Path(temporary) / "manifest"
            dataset_root.mkdir()
            _write_features_csv(dataset_root, self.FEATURE)
            archive, video_member, timestamps_member, video_bytes, timestamps_bytes = (
                _write_camera_archive(
                    dataset_root,
                    feature=self.FEATURE,
                    clip_id=self.CLIP_ID,
                )
            )

            self._run_prepare(dataset_root, output_root)

            manifest = output_root / "pai_nvc_video_manifest.jsonl"
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["clip_id"], self.CLIP_ID)
            self.assertEqual(record["camera_feature"], self.FEATURE)
            self.assertEqual(record["video_zip_path"], str(archive.resolve()))
            self.assertEqual(record["video_member"], video_member)
            self.assertEqual(record["timestamps_zip_path"], str(archive.resolve()))
            self.assertEqual(record["timestamps_member"], timestamps_member)

            with zipfile.ZipFile(archive) as handle:
                video_info = handle.getinfo(video_member)
                timestamp_info = handle.getinfo(timestamps_member)
            video_offset = _member_payload_offset(archive, video_info)
            timestamp_offset = _member_payload_offset(archive, timestamp_info)
            self.assertEqual(record["video_payload_offset"], video_offset)
            self.assertEqual(record["video_size"], len(video_bytes))
            self.assertEqual(record["timestamps_payload_offset"], timestamp_offset)
            self.assertEqual(record["timestamps_size"], len(timestamps_bytes))

            with archive.open("rb") as stream:
                stream.seek(video_offset)
                self.assertEqual(stream.read(len(video_bytes)), video_bytes)
                stream.seek(timestamp_offset)
                self.assertEqual(stream.read(len(timestamps_bytes)), timestamps_bytes)

            # FFmpeg's subfile protocol must receive the member's exact byte
            # range, not a URL to the whole archive or to an extracted copy.
            subfile_url = record["video_path"]
            self.assertIn(f"start,{video_offset}", subfile_url)
            self.assertIn(f"end,{video_offset + len(video_bytes)}", subfile_url)
            self.assertTrue(subfile_url.endswith(f":{archive.resolve()}"))
            self.assertFalse(list(output_root.rglob("*.mp4")))

    def test_zip_subfile_manifest_rejects_deflated_members(self) -> None:
        for compressed_member in ("video", "timestamps"):
            with (
                self.subTest(compressed_member=compressed_member),
                tempfile.TemporaryDirectory() as temporary,
            ):
                dataset_root = Path(temporary) / "dataset"
                output_root = Path(temporary) / "manifest"
                dataset_root.mkdir()
                _write_features_csv(dataset_root, self.FEATURE)
                _write_camera_archive(
                    dataset_root,
                    feature=self.FEATURE,
                    clip_id=self.CLIP_ID,
                    video_compression=(
                        zipfile.ZIP_DEFLATED
                        if compressed_member == "video"
                        else zipfile.ZIP_STORED
                    ),
                    timestamp_compression=(
                        zipfile.ZIP_DEFLATED
                        if compressed_member == "timestamps"
                        else zipfile.ZIP_STORED
                    ),
                )

                with self.assertRaisesRegex(ValueError, "ZIP_STORED"):
                    self._run_prepare(dataset_root, output_root)


class NvcZipTimestampTests(unittest.TestCase):
    def test_timestamp_parquet_is_read_from_zip_member_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "camera.zip"
            member = "clip.camera.timestamps.parquet"
            payload = b"small-parquet-payload"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
                archive.writestr(member, payload)

            parquet_inputs: list[bytes] = []

            def read_parquet(source):
                self.assertIsInstance(source, io.BytesIO)
                parquet_inputs.append(source.read())
                return _TimestampFrame(_Array([100, 200, 300]))

            module = _load_nvc_pai_module(read_parquet)
            dataset = object.__new__(module.NvcPAIDataset)
            dataset._timestamp_cache = {}
            record = {
                "timestamps_zip_path": str(archive_path),
                "timestamps_member": member,
            }

            first = dataset._timestamps_for_record(record)
            second = dataset._timestamps_for_record(record)

            self.assertEqual(first.values, [100, 200, 300])
            self.assertIs(first, second)
            self.assertEqual(parquet_inputs, [payload])
            self.assertIn(("zip", str(archive_path), member), dataset._timestamp_cache)

    def test_frame_ids_choose_the_frame_at_or_before_each_request(self) -> None:
        module = _load_nvc_pai_module(lambda _source: None)

        module.np.asarray = lambda values, *, dtype: _Array(values)
        module.np.any = lambda values: any(values)
        module.np.searchsorted = lambda timestamps, requested, *, side: _Array(
            [
                bisect.bisect_right(timestamps.values, request)
                if side == "right"
                else bisect.bisect_left(timestamps.values, request)
                for request in requested.values
            ]
        )

        class _AbsoluteTimestamps:
            def min(self):
                return 0

            def __sub__(self, _other):
                return self

            def float(self):
                return self

            def __mul__(self, _other):
                return self

        module.torch.from_numpy = lambda _timestamps: object()
        module.torch.stack = lambda _timestamps: _AbsoluteTimestamps()
        module.torch.tensor = lambda values, *, dtype: tuple(values)

        timestamps = _Array([0, 900_000, 2_100_000, 2_900_000, 3_100_000])
        dataset = object.__new__(module.NvcPAIDataset)
        dataset.num_frames = 4
        dataset.time_step = 1.0
        dataset._timestamp_cache = {}
        dataset._manifest = {
            ("clip-001", feature): {"video_path": f"subfile:{feature}"}
            for feature in module.CAMERA_FEATURES
        }
        dataset._timestamps_for_record = lambda _record: timestamps

        paths, frame_ids, camera_indices, _relative = dataset._frame_ids("clip-001", 3_000_000)

        self.assertEqual(frame_ids, ((0, 1, 1, 3),) * len(module.CAMERA_FEATURES))
        self.assertEqual(paths, tuple(f"subfile:{feature}" for feature in module.CAMERA_FEATURES))
        self.assertEqual(camera_indices, (0, 1, 2, 6))


if __name__ == "__main__":
    unittest.main()
