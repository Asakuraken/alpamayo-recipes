#!/usr/bin/env python3
"""Prepare PAI camera videos for ACCV-Lab's GOP decoder.

PAI stores each camera clip and its timestamp parquet inside a chunk ZIP.  By
default this tool performs the original lossless extraction and writes a JSONL
manifest that maps a ``(clip_id, camera_feature)`` pair to the extracted MP4
and timestamp files.

``--mode zip-subfile`` avoids materializing the MP4s.  It writes FFmpeg's
``subfile`` URLs that address the raw MP4 byte range inside each source ZIP,
plus the ZIP/member metadata needed to read the timestamp parquet.  This mode
requires both members to be ``ZIP_STORED``: an FFmpeg subfile URL cannot
inflate a compressed or encrypted ZIP member.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import struct
import zipfile


_LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50


def _camera_features(features_csv: Path) -> dict[str, dict[str, str]]:
    with features_csv.open(newline="") as stream:
        rows = csv.DictReader(stream)
        return {
            row["feature"]: row
            for row in rows
            if row["feature"].startswith("camera_") and row["chunk_path"].endswith(".zip")
        }


def _member_names(clip_id: str, feature: str) -> tuple[str, str]:
    stem = f"{clip_id}.{feature}"
    return f"{stem}.mp4", f"{stem}.timestamps.parquet"


def _extract_member(
    archive: zipfile.ZipFile, member: str, destination: Path, overwrite: bool
) -> None:
    info = archive.getinfo(member)
    if destination.exists():
        if not overwrite and destination.stat().st_size == info.file_size:
            return
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists but does not match {member}; "
                "pass --overwrite to replace it"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _zip_member_payload_offset(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> int:
    """Return the member's raw-data offset using its local ZIP file header.

    ``ZipInfo.header_offset`` points at the start of the local header, not the
    member payload.  The variable-length filename and extra fields in that
    header must be included; the central-directory values alone are not a
    reliable substitute.
    """

    if archive.fp is None:
        raise ValueError(f"Cannot read local header for {info.filename}: archive is closed")
    archive.fp.seek(info.header_offset)
    header = archive.fp.read(_LOCAL_FILE_HEADER.size)
    if len(header) != _LOCAL_FILE_HEADER.size:
        raise ValueError(f"Truncated local header for ZIP member {info.filename}")
    (
        signature,
        _version_needed,
        flags,
        compression_method,
        _modified_time,
        _modified_date,
        _crc32,
        _compressed_size,
        _uncompressed_size,
        filename_size,
        extra_size,
    ) = _LOCAL_FILE_HEADER.unpack(header)
    if signature != _LOCAL_FILE_HEADER_SIGNATURE:
        raise ValueError(
            f"Invalid local header signature for ZIP member {info.filename}: "
            f"0x{signature:08x}"
        )
    if compression_method != info.compress_type:
        raise ValueError(
            f"Local header compression method for {info.filename} "
            f"({compression_method}) does not match the central directory "
            f"({info.compress_type})"
        )
    if flags & 0x1:
        raise ValueError(f"ZIP member {info.filename} is encrypted")
    return info.header_offset + _LOCAL_FILE_HEADER.size + filename_size + extra_size


def _stored_member_metadata(
    archive: zipfile.ZipFile, archive_path: Path, member: str
) -> tuple[zipfile.ZipInfo, int]:
    """Validate a member can be exposed as a raw FFmpeg subfile range."""

    info = archive.getinfo(member)
    if info.compress_type != zipfile.ZIP_STORED:
        compression_name = zipfile.compressor_names.get(info.compress_type, str(info.compress_type))
        raise ValueError(
            f"{archive_path}: {member} uses {compression_name}; "
            "zip-subfile mode requires ZIP_STORED members"
        )
    if info.flag_bits & 0x1:
        raise ValueError(f"{archive_path}: ZIP member {member} is encrypted")
    if info.compress_size != info.file_size:
        raise ValueError(
            f"{archive_path}: stored member {member} has compressed size "
            f"{info.compress_size}, expected {info.file_size}"
        )
    return info, _zip_member_payload_offset(archive, info)


def _subfile_url(archive_path: Path, payload_offset: int, payload_size: int) -> str:
    """Build an FFmpeg URL that exposes one stored ZIP member as a file."""

    if payload_offset < 0 or payload_size < 0:
        raise ValueError("ZIP member offsets and sizes must be non-negative")
    payload_end = payload_offset + payload_size
    # This is FFmpeg's documented subfile syntax.  The colon before an
    # absolute path is part of the protocol's option separator, not a typo.
    return f"subfile,,start,{payload_offset},end,{payload_end},,:{archive_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--chunks",
        default=None,
        help="Comma-separated PAI chunk IDs; default processes every camera archive found.",
    )
    parser.add_argument(
        "--mode",
        choices=("extract", "zip-subfile"),
        default="extract",
        help=(
            "extract (default): materialize MP4/parquet files under output_root; "
            "zip-subfile: leave source ZIPs in place and write FFmpeg subfile URLs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing extracted files (only relevant in extract mode).",
    )
    args = parser.parse_args()

    source_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    selected_chunks = (
        {int(value) for value in args.chunks.split(",") if value.strip()}
        if args.chunks
        else None
    )
    features = _camera_features(source_root / "features.csv")
    records: list[dict[str, object]] = []

    for feature, metadata in sorted(features.items()):
        pattern = metadata["chunk_path"].replace("{chunk_id:04d}", "*")
        for archive_path in sorted(source_root.glob(pattern)):
            # Camera archive filenames end in ``.chunk_XXXX.zip``.
            chunk = int(archive_path.stem.rsplit("_", 1)[-1])
            if selected_chunks is not None and chunk not in selected_chunks:
                continue
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    suffix = f".{feature}.mp4"
                    if not member.endswith(suffix):
                        continue
                    clip_id = member[: -len(suffix)]
                    mp4_member, timestamps_member = _member_names(clip_id, feature)
                    # A video without timestamps cannot be sampled equivalently to
                    # the baseline reader, so fail before writing a partial record.
                    if args.mode == "extract":
                        archive.getinfo(timestamps_member)
                        feature_dir = output_root / feature / f"chunk_{chunk:04d}"
                        mp4_path = feature_dir / mp4_member
                        timestamps_path = feature_dir / timestamps_member
                        _extract_member(archive, mp4_member, mp4_path, args.overwrite)
                        _extract_member(
                            archive, timestamps_member, timestamps_path, args.overwrite
                        )
                        records.append(
                            {
                                "clip_id": clip_id,
                                "camera_feature": feature,
                                "chunk_id": str(chunk),
                                "video_path": str(mp4_path),
                                "timestamps_path": str(timestamps_path),
                            }
                        )
                        continue

                    mp4_info, mp4_offset = _stored_member_metadata(
                        archive, archive_path, mp4_member
                    )
                    timestamps_info, timestamps_offset = _stored_member_metadata(
                        archive, archive_path, timestamps_member
                    )
                    records.append(
                        {
                            "clip_id": clip_id,
                            "camera_feature": feature,
                            "chunk_id": str(chunk),
                            "video_path": _subfile_url(
                                archive_path, mp4_offset, mp4_info.file_size
                            ),
                            "video_zip_path": str(archive_path),
                            "video_member": mp4_member,
                            "video_payload_offset": mp4_offset,
                            "video_size": mp4_info.file_size,
                            "timestamps_zip_path": str(archive_path),
                            "timestamps_member": timestamps_member,
                            "timestamps_payload_offset": timestamps_offset,
                            "timestamps_size": timestamps_info.file_size,
                        }
                    )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "pai_nvc_video_manifest.jsonl"
    with manifest.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    if args.mode == "extract":
        print(f"Extracted or verified {len(records)} camera videos; manifest: {manifest}")
    else:
        print(
            f"Wrote {len(records)} ZIP-subfile camera video entries "
            f"(no MP4s extracted); manifest: {manifest}"
        )


if __name__ == "__main__":
    main()
