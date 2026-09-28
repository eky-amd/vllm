# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm platform defaults"
)


def _defaults(custom_ops: list[str], quantization: str | None) -> list[str]:
    from vllm.platforms.rocm import RocmPlatform

    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(custom_ops=list(custom_ops)),
        model_config=SimpleNamespace(quantization=quantization),
    )
    RocmPlatform.apply_config_platform_defaults(cfg)
    return cfg.compilation_config.custom_ops


def test_on_by_default_for_unquantized_models():
    assert "+rms_norm_gated" in _defaults(["none"], quantization=None)


@pytest.mark.parametrize("quantization", ["fp8", "quark", "compressed-tensors"])
def test_quantized_models_keep_the_native_decomposition(quantization):
    assert "+rms_norm_gated" not in _defaults(["none"], quantization)


def test_user_opt_out_is_respected():
    assert "+rms_norm_gated" not in _defaults(["none", "-rms_norm_gated"], None)


def test_no_duplicate_when_already_enabled():
    assert _defaults(["none", "+rms_norm_gated"], None).count("+rms_norm_gated") == 1


def test_missing_model_config_is_tolerated():
    from vllm.platforms.rocm import RocmPlatform

    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(custom_ops=["none"]), model_config=None
    )
    RocmPlatform.apply_config_platform_defaults(cfg)
    assert "+rms_norm_gated" not in cfg.compilation_config.custom_ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("num_tokens", [7, 64])
@pytest.mark.parametrize("transposed", [False, True])
@torch.inference_mode()
def test_enabled_op_compiles_to_the_triton_kernel(num_tokens: int, transposed: bool):
    """With the op enabled, torch.compile must trace the FLA Triton path
    without a graph break and match the native decomposition, also when the
    input is a strided view (the fake impl must promise the contiguous output
    the kernel produces, or inductor lays out the consumer wrongly)."""
    heads, head_dim = 48, 128
    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["none", "+rms_norm_gated"])
    )
    with set_current_vllm_config(config):
        norm = RMSNormGated(
            head_dim,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            activation="silu",
            device="cuda",
        )
        assert norm.enabled()
        torch.manual_seed(0)
        x = torch.randn(
            num_tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        if transposed:
            x = torch.randn(
                heads, num_tokens, head_dim, device="cuda", dtype=torch.bfloat16
            ).transpose(0, 1)
        z = torch.randn(x.shape, device="cuda", dtype=torch.bfloat16)
        ref = norm.forward_native(x.clone(), z.clone())
        # the consumer's layout must be what the real kernel produces
        proj = torch.randn(head_dim * heads, 8, device="cuda", dtype=torch.bfloat16)

        def f(x, z):
            return norm(x, z).flatten(-2) @ proj

        compiled = torch.compile(f, fullgraph=True)
        out = compiled(x, z)
        ref = ref.flatten(-2) @ proj
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_fusion_matcher_traces_the_enabled_op_without_launching_a_kernel():
    """RocmAiterRMSNormQuantFusionPass builds its gated-norm pattern from
    MatcherRMSNormGated on fake tensors. With the op enabled that must produce
    the opaque op node, not a Triton launch on fake pointers (which faulted
    the GPU at server start-up)."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    from vllm.compilation.passes.fusion.matcher_utils import MatcherRMSNormGated

    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["none", "+rms_norm_gated"])
    )
    with set_current_vllm_config(config):
        matcher = MatcherRMSNormGated(1e-6, enabled=True)
        x, z, w = matcher.inputs()
        with FakeTensorMode() as mode:
            out = matcher(*(mode.from_tensor(t) for t in (x, z, w)))
        assert out.shape == x.shape
        graph = make_fx(matcher, tracing_mode="fake")(x, z, w).graph
    targets = {str(n.target) for n in graph.nodes if n.op == "call_function"}
    assert any("fla_rms_norm_gated" in t for t in targets), targets
