from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


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


def _load_module(path: Path, name: str, stubs: dict[str, types.ModuleType]):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _temporary_modules({**stubs, name: module}):
        spec.loader.exec_module(module)
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_prefetcher_module():
    class _Tensor:
        def __init__(self, *, is_cuda: bool = True) -> None:
            self.is_cuda = is_cuda
            self.recorded_streams: list[object] = []

        def record_stream(self, stream: object) -> None:
            self.recorded_streams.append(stream)

    torch = types.ModuleType("torch")
    torch.Tensor = _Tensor
    torch.cuda = types.SimpleNamespace(set_device=lambda _device: None)

    class _NvcGopRequest:
        def __init__(self, video_paths, frame_ids, camera_features, gop_refs) -> None:
            self.video_paths = video_paths
            self.frame_ids = frame_ids
            self.camera_features = camera_features
            self.gop_refs = gop_refs

    nvc_gop = types.ModuleType("alpamayo.data.nvc_gop")
    nvc_gop.NVC_GOP_REQUEST_KEY = "_nvc_gop_request"
    nvc_gop.NvcGopRequest = _NvcGopRequest
    nvc_gop.calculate_store_capacity = lambda **_kwargs: 1
    nvc_gop.make_store_id = lambda _global_rank=0: 0

    return _load_module(
        ROOT / "src" / "alpamayo" / "data" / "nvc_gop_prefetcher.py",
        "_nvc_gop_prefetcher_test",
        {
            "torch": torch,
            "alpamayo": _package("alpamayo"),
            "alpamayo.data": _package("alpamayo.data"),
            "alpamayo.data.nvc_gop": nvc_gop,
        },
    )


def _load_nvc_pai_processor_module():
    torch = types.ModuleType("torch")

    nvc_gop = types.ModuleType("alpamayo.data.nvc_gop")
    nvc_gop.NVC_GOP_REQUEST_KEY = "_nvc_gop_request"

    return _load_module(
        ROOT / "src" / "alpamayo" / "data" / "nvc_pai_processor.py",
        "_nvc_pai_processor_test",
        {
            "torch": torch,
            "alpamayo": _package("alpamayo"),
            "alpamayo.data": _package("alpamayo.data"),
            "alpamayo.data.nvc_gop": nvc_gop,
        },
    )


def _load_nvc_gop_module():
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object

    return _load_module(
        ROOT / "src" / "alpamayo" / "data" / "nvc_gop.py",
        "_nvc_gop_test",
        {
            "numpy": numpy,
            "alpamayo": _package("alpamayo"),
            "alpamayo.data": _package("alpamayo.data"),
        },
    )


def _load_qwen_processor_module():
    torch = types.ModuleType("torch")
    torch.Tensor = object

    class _Tokenizer:
        def add_tokens(self, tokens, special_tokens=False):
            del special_tokens
            return len(tokens)

        def convert_tokens_to_ids(self, token):
            return hash(token)

    class _AutoProcessor:
        build_count = 0

        @classmethod
        def from_pretrained(cls, _path, **_kwargs):
            cls.build_count += 1
            return types.SimpleNamespace(tokenizer=_Tokenizer())

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = _AutoProcessor

    chat_template = types.ModuleType("alpamayo.chat_template")
    chat_template.get_template = lambda _version: object()

    base_model = types.ModuleType("alpamayo_r1.models.base_model")
    base_model.SPECIAL_TOKENS = {"special": "<special>"}
    base_model.TRAJ_TOKEN = {"start": "<traj_start>"}

    label_mask = types.ModuleType("alpamayo.utils.get_label_mask")
    label_mask.get_label_mask = lambda **_kwargs: None
    label_mask.get_role_eos_mask = lambda **_kwargs: None

    module = _load_module(
        ROOT / "src" / "alpamayo" / "processor" / "qwen_processor.py",
        "_qwen_processor_accv_test",
        {
            "torch": torch,
            "transformers": transformers,
            "alpamayo": _package("alpamayo"),
            "alpamayo.chat_template": chat_template,
            "alpamayo.utils": _package("alpamayo.utils"),
            "alpamayo.utils.get_label_mask": label_mask,
            "alpamayo_r1": _package("alpamayo_r1"),
            "alpamayo_r1.models": _package("alpamayo_r1.models"),
            "alpamayo_r1.models.base_model": base_model,
        },
    )
    return module, _AutoProcessor


class _Owner:
    device = "cuda:0"

    def __init__(self) -> None:
        self.materialized: list[int] = []
        self.fifth_materialized = threading.Event()

    def _submit(self, raw_batch: int) -> int:
        return raw_batch

    def _collect_and_clone(self, pending: int) -> list[int]:
        return [pending]

    def _materialize(self, pending: int, _samples: list[int]):
        self.materialized.append(pending)
        if len(self.materialized) == 5:
            self.fifth_materialized.set()
        return pending, f"ready-{pending}"


class AccvPipelineTests(unittest.TestCase):
    def test_reused_qwen_collator_builds_auto_processor_once(self) -> None:
        module, auto_processor = _load_qwen_processor_module()
        model_config = types.SimpleNamespace(
            vlm_name_or_path="fake-model",
            traj_vocab_size=None,
            min_pixels=None,
            max_pixels=None,
        )

        qwen_processor = module.QwenProcessor(
            vlm_name_or_path=model_config.vlm_name_or_path,
            traj_vocab_size=model_config.traj_vocab_size,
            min_pixels=model_config.min_pixels,
            max_pixels=model_config.max_pixels,
        )
        collate_fn = qwen_processor.collate_fn

        self.assertIs(collate_fn.__self__, qwen_processor)
        self.assertIs(qwen_processor.processor, qwen_processor.processor)
        self.assertEqual(auto_processor.build_count, 1)

    def test_worker_prefetches_a_bounded_accumulation_window_in_order(self) -> None:
        module = _load_prefetcher_module()
        owner = _Owner()
        worker = module._DecodeTransformWorker(owner, iter(range(6)), capacity=4)
        worker.start()

        self.assertTrue(owner.fifth_materialized.wait(timeout=2))
        self.assertEqual(worker.results.qsize(), 4)
        for expected in range(6):
            self.assertEqual(worker.wait(), (expected, f"ready-{expected}"))
        self.assertIsNone(worker.wait())

        worker.request_stop()
        worker.join()

    def test_worker_propagates_background_failures(self) -> None:
        module = _load_prefetcher_module()

        class _FailingOwner(_Owner):
            def _submit(self, raw_batch: int) -> int:
                if raw_batch == 1:
                    raise ValueError("decode failed")
                return raw_batch

        worker = module._DecodeTransformWorker(_FailingOwner(), iter(range(2)), capacity=2)
        worker.start()
        self.assertEqual(worker.wait(), (0, "ready-0"))
        with self.assertRaisesRegex(ValueError, "decode failed"):
            worker.wait()

        worker.request_stop()
        worker.join()

    def test_worker_can_stop_while_result_queue_is_full(self) -> None:
        module = _load_prefetcher_module()
        owner = _Owner()
        worker = module._DecodeTransformWorker(owner, iter(range(100)), capacity=4)
        worker.start()

        self.assertTrue(owner.fifth_materialized.wait(timeout=2))
        worker.request_stop()
        worker.join()
        self.assertIsNone(worker.thread)

    def test_submit_passes_shared_gop_views_directly_to_decoder(self) -> None:
        module = _load_prefetcher_module()
        shared = types.SimpleNamespace(nbytes=8)
        ref = types.SimpleNamespace(
            first_frame_id=0,
            gop_len=4,
            shm_name="gop-0",
            data_size=8,
        )

        class _Store:
            def get_batch(self, refs):
                self.refs = refs
                return [shared]

        class _Decoder:
            def DecodeFromGOPListRGB(self, bundles, paths, frame_ids, flag):
                self.bundles = bundles
                self.paths = paths
                self.frame_ids = frame_ids
                self.flag = flag

        store = _Store()
        decoder = _Decoder()
        owner = object.__new__(module.NvcGopBatchPrefetcher)
        owner.max_grouped_frames = 4
        owner._store = store
        owner._decoder = decoder
        request = module.NvcGopRequest(
            video_paths=("video.mp4",),
            frame_ids=((3,),),
            camera_features=("front",),
            gop_refs=((ref,),),
        )

        pending = owner._submit_impl(
            {"inputs": {module.NVC_GOP_REQUEST_KEY: [request]}}
        )

        self.assertEqual(store.refs, [ref])
        self.assertIs(decoder.bundles[0][0], shared)
        self.assertFalse(hasattr(pending, "numpy_datas"))

    def test_persistent_iterator_defers_raw_worker_shutdown_until_final_close(self) -> None:
        module = _load_prefetcher_module()

        class _RawIterator:
            def __init__(self) -> None:
                self.close_count = 0

            def close(self) -> None:
                self.close_count += 1

        class _DataLoaderIterator:
            def __init__(self) -> None:
                self.shutdown_count = 0

            def _shutdown_workers(self) -> None:
                self.shutdown_count += 1

        class _AccelerateDataLoader:
            def __init__(self, base_dataloader) -> None:
                self.base_dataloader = base_dataloader

        class _Worker:
            def __init__(self) -> None:
                self.stop_count = 0
                self.join_count = 0

            def request_stop(self) -> None:
                self.stop_count += 1

            def join(self) -> None:
                self.join_count += 1

        raw_iterator = _RawIterator()
        worker = _Worker()
        dataloader_iterator = _DataLoaderIterator()
        dataloader = _AccelerateDataLoader(
            types.SimpleNamespace(
                _iterator=dataloader_iterator,
                persistent_workers=True,
            )
        )
        iterator = object.__new__(module._NvcGopIterator)
        iterator.owner = types.SimpleNamespace(
            dataloader=dataloader,
            _active_iterator=True,
        )
        iterator.raw_iterator = raw_iterator
        iterator.worker = worker
        iterator._closed = False
        iterator._raw_workers_shutdown = False
        iterator._raw_iterator_closed = False

        iterator.close()

        self.assertEqual(worker.stop_count, 1)
        self.assertEqual(worker.join_count, 1)
        self.assertEqual(raw_iterator.close_count, 1)
        self.assertEqual(dataloader_iterator.shutdown_count, 0)

        iterator.close(shutdown_workers=True)

        self.assertEqual(raw_iterator.close_count, 1)
        self.assertEqual(dataloader_iterator.shutdown_count, 1)

    def test_vla_collation_releases_raw_nvdec_frames(self) -> None:
        module = _load_nvc_pai_processor_module()

        class _Frames:
            shape = (2, 3, 4, 5)

            def reshape(self, *shape):
                self.reshape_shape = shape
                return self

            def permute(self, *axes):
                self.permute_axes = axes
                return self

        frames = _Frames()
        preprocessed_frames: list[object] = []

        def preprocess(*, data):
            preprocessed_frames.append(data["image_frames"])
            return {"pixel_values": "ready"}

        ready_batches: list[list[dict[str, object]]] = []
        processor = module.NvcPAIBatchProcessor(
            types.SimpleNamespace(vla_preprocess_func=preprocess),
            lambda ready: ready_batches.append(ready) or {"ready": ready},
        )
        request = types.SimpleNamespace(
            camera_features=("front",), frame_ids=((1, 2),)
        )

        result = processor.materialize_batch(
            {
                "inputs": {
                    "_nvc_samples": [{"sample": "value"}],
                    module.NVC_GOP_REQUEST_KEY: [request],
                }
            },
            [frames],
        )

        self.assertEqual(result["ready"][0]["tokenized_data"], {"pixel_values": "ready"})
        self.assertEqual(preprocessed_frames, [frames])
        self.assertNotIn("image_frames", ready_batches[0][0])

    def test_gop_cache_is_worker_local_and_store_capacity_counts_one_main_batch(self) -> None:
        module = _load_nvc_gop_module()
        self.assertEqual(
            module.calculate_store_capacity(
                batch_size=1,
                num_workers=2,
                prefetch_factor=2,
                num_cameras=4,
                num_frames=4,
            ),
            80,
        )

        ref = types.SimpleNamespace(first_frame_id=0, gop_len=1, shm_name="gop-0")

        class _Store:
            def lookup(self, *_args):
                return None

            def put(self, *_args):
                return ref

        class _Demuxer:
            def GetGOPList(self, _paths, _frame_ids, *, useGOPCache):
                self.use_gop_cache = useGOPCache
                return [(object(), [0], [1])]

        demuxer = _Demuxer()
        module.own_serialized_gop_bundle = lambda _data, **_kwargs: object()
        request = module.NvcGopRequest(
            video_paths=("video.mp4",),
            frame_ids=((3,),),
            camera_features=("front",),
        )

        module.materialize_gop_request(request, store=_Store(), demuxer=demuxer)

        self.assertTrue(demuxer.use_gop_cache)

    def test_nested_cuda_tensors_are_recorded_on_consumer_stream_once(self) -> None:
        module = _load_prefetcher_module()
        stream = object()
        cuda_tensor = module.torch.Tensor()
        second_cuda_tensor = module.torch.Tensor()
        cpu_tensor = module.torch.Tensor(is_cuda=False)
        batch = {
            "inputs": [cuda_tensor, (second_cuda_tensor, {"alias": cuda_tensor})],
            "cpu": cpu_tensor,
        }

        module._record_cuda_tensors(batch, stream)

        self.assertEqual(cuda_tensor.recorded_streams, [stream])
        self.assertEqual(second_cuda_tensor.recorded_streams, [stream])
        self.assertEqual(cpu_tensor.recorded_streams, [])

    def test_materialize_records_source_tensors_on_transform_stream(self) -> None:
        module = _load_prefetcher_module()
        stream = object()
        sample = module.torch.Tensor()

        @contextmanager
        def use_stream(selected_stream):
            self.assertIs(selected_stream, stream)
            yield

        class _Event:
            def record(self, selected_stream):
                self.recorded_stream = selected_stream

        module.torch.cuda.stream = use_stream
        module.torch.cuda.Event = _Event
        owner = object.__new__(module.NvcGopBatchPrefetcher)
        owner._transform_stream = stream
        owner.processor = types.SimpleNamespace(
            materialize_batch=lambda batch, _samples: batch
        )
        pending = types.SimpleNamespace(batch={"inputs": {}})

        batch, ready_event = owner._materialize(pending, [sample])

        self.assertEqual(batch, {"inputs": {}})
        self.assertIs(ready_event.recorded_stream, stream)
        self.assertEqual(sample.recorded_streams, [stream])

    def test_iterator_waits_then_records_batch_on_consumer_stream(self) -> None:
        module = _load_prefetcher_module()
        events: list[str] = []

        class _ConsumerStream:
            def wait_event(self, ready_event):
                events.append(f"wait:{ready_event}")

        stream = _ConsumerStream()
        tensor = module.torch.Tensor()
        tensor.record_stream = lambda selected_stream: events.append(
            "record" if selected_stream is stream else "wrong-stream"
        )
        module.torch.cuda.current_stream = lambda _device: stream
        iterator = object.__new__(module._NvcGopIterator)
        iterator._closed = True
        iterator.owner = types.SimpleNamespace(device="cuda:0")
        iterator.worker = types.SimpleNamespace(
            wait=lambda: ({"inputs": {"pixel_values": tensor}}, "ready")
        )

        batch = next(iterator)

        self.assertIs(batch["inputs"]["pixel_values"], tensor)
        self.assertEqual(events, ["wait:ready", "record"])


if __name__ == "__main__":
    unittest.main()
