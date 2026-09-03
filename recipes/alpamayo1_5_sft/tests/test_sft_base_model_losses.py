# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA


def _next_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return TrainableReasoningVLA._compute_next_token_loss(
        None,
        SimpleNamespace(logits=logits),
        labels,
        labels_mask,
        token_mask,
    )


def test_next_token_loss_empty_mask_is_zero_and_backpropagates() -> None:
    logits = torch.randn(1, 4, 8, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4]])
    labels_mask = torch.zeros_like(labels, dtype=torch.bool)

    loss = _next_token_loss(logits, labels, labels_mask)

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_next_token_loss_token_mask_does_not_mutate_full_logits() -> None:
    logits = torch.randn(1, 4, 8)
    logits_before = logits.clone()
    labels = torch.tensor([[1, 2, 3, 4]])
    labels_mask = torch.ones_like(labels, dtype=torch.bool)
    token_mask = torch.tensor([True, True, True, True, True, False, False, False])

    _next_token_loss(logits, labels, labels_mask, token_mask)

    torch.testing.assert_close(logits, logits_before)


class _RecordingVLM:
    def __init__(self) -> None:
        self.call_kwargs: dict[str, torch.Tensor] | None = None

    def __call__(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        self.call_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        return SimpleNamespace(
            logits=torch.randn(
                *input_ids.shape,
                16,
                device=input_ids.device,
                requires_grad=True,
            ),
            loss=None,
        )


def test_forward_uses_only_the_masked_losses() -> None:
    vlm = _RecordingVLM()
    model = SimpleNamespace(
        vlm=vlm,
        config=SimpleNamespace(traj_vocab_size=2),
        future_token_start_idx=10,
        special_token_ids={"traj_future_start": 14, "traj_future_end": 15},
    )
    model.fuse_traj_tokens = lambda input_ids, _traj_data: input_ids
    model._compute_next_token_loss = lambda outputs, labels, labels_mask: _next_token_loss(
        outputs.logits, labels, labels_mask
    )

    output = TrainableReasoningVLA.forward(
        model,
        {"input_ids": torch.tensor([[1, 10, 11, 2]])},
    )

    assert vlm.call_kwargs is not None
    assert "labels" not in vlm.call_kwargs
    assert output.loss is not None
    assert torch.isfinite(output.loss)
