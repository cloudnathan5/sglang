# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sglang project
"""Dense NVFP4 with unquantized activations (W4A16) for compressed-tensors.

The kernel path already exists: ModelOpt's ``ModelOptNvFp4A16LinearMethod``
serves NVFP4 weight-only checkpoints through the FP4 Marlin kernels. It was only
reachable for ModelOpt-format checkpoints, so a compressed-tensors
``nvfp4-pack-quantized`` checkpoint with ``input_activations: null`` had no
scheme and fell through to the W8A8 detectors, which dereference a ``None``
``input_quant``.

This scheme reuses those Marlin kernels and differs from
``CompressedTensorsW4A4Fp4`` only on the activation side: no ``input_global_scale``
is registered or expected, and activations stay in fp16/bf16.
"""

import logging
from collections.abc import Callable
from typing import Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_nvfp4_layer_for_marlin,
)

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A16Fp4"]

# NVFP4 packs 16 FP4 values against one FP8 block scale; the Marlin path assumes
# that grouping, so a checkpoint quantized with any other group size cannot be
# served here.
NVFP4_GROUP_SIZE = 16


class _MarlinGroupSizeShim:
    """`prepare_nvfp4_layer_for_marlin` reads ``layer.quant_config.group_size``.

    The compressed-tensors path has no per-layer quant_config object, so supply
    the one attribute that helper needs rather than threading the whole config
    through.
    """

    group_size = NVFP4_GROUP_SIZE


class CompressedTensorsW4A16Fp4(CompressedTensorsLinearScheme):
    def __init__(self):
        self.group_size = NVFP4_GROUP_SIZE

    @classmethod
    def get_min_capability(cls) -> int:
        # FP4 Marlin is the SM80+ weight-only path; unlike W4A4 this does not
        # need FP4 tensor cores, since activations stay in fp16/bf16.
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"NVFP4 W4A16 requires the input feature size to be a multiple "
                f"of {self.group_size}, got {input_size_per_partition}."
            )

        # Two FP4 values per byte.
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # Deliberately no input_global_scale: activations are not quantized.

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fused parallel layers (qkv_proj, gate_up_proj) load one global scale
        # per constituent projection. Marlin dequantizes with a single scalar,
        # so collapse to the max; warn when they disagree, since that costs
        # accuracy on whichever projection is scaled down.
        if torch.unique(layer.weight_global_scale).numel() != 1:
            logger.warning_once(
                "NVFP4 W4A16: weight_global_scale differs across fused parallel "
                "layers; collapsing to max. Accuracy may be degraded. Prefer a "
                "checkpoint quantized with a shared global scale for q/k/v and "
                "gate/up."
            )
        # compressed-tensors stores the PRE-quant multiplier
        # (global_scale = 448*6/amax, applied before rounding); the Marlin path
        # inherited ModelOpt semantics, where weight_scale_2 is the DEQUANT
        # multiplier. They are reciprocals -- vLLM's compressed-tensors scheme
        # makes the same 1/x conversion before its kernels see the value.
        layer.weight_global_scale = Parameter(
            1.0 / layer.weight_global_scale.max().to(torch.float32),
            requires_grad=False,
        )

        # The Marlin helpers operate on `weight`; compressed-tensors names the
        # packed tensor `weight_packed`.
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

        layer.quant_config = _MarlinGroupSizeShim()
        prepare_nvfp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
