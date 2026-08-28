# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER fused_qk_rope_concat_and_cache_mla (VLLM_ROCM_USE_AITER_FUSED_QK_ROPE_CACHE_MLA)
vs the un-fused reference: DeepSeek YaRN rope -> concat_and_cache_mla -> cat + static fp8 quant."""
import pytest
import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_rocm() or not rocm_aiter_ops.is_enabled(),
    reason="ROCm AITER only",
)
def test_fused_qk_rope_concat_and_cache_mla_matches_reference():
    with set_current_vllm_config(VllmConfig()):
        _run()


def _run():
    torch.manual_seed(0); dev = "cuda"

    B, N, L, P, BS, NB = 64, 16, 512, 64, 64, 64  # tokens, heads, kv_lora, pe, block_size, blocks
    fp8 = current_platform.fp8_dtype()
    rope_scaling = {"beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 1.0, "mscale_all_dim": 1.0,
                    "original_max_position_embeddings": 4096, "rope_type": "deepseek_yarn"}
    rope = get_rope(P, max_position=4096 * 40, is_neox_style=False, rope_parameters={"rope_theta": 10000, **rope_scaling},
                    dtype=torch.bfloat16).to(dev)
    ql_nope_t = torch.randn(N, B, L, device=dev, dtype=torch.bfloat16) * 0.5   # (N,B,L) storage like the bmm output
    ql_nope = ql_nope_t.transpose(0, 1)                                          # (B,N,L) transposed view
    q_full = torch.randn(B, N, 128 + P, device=dev, dtype=torch.bfloat16)
    q_pe = q_full[..., 128:]                                                     # strided view like q.split()
    kv_c = torch.randn(B, L, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(B, 1, P, device=dev, dtype=torch.bfloat16)
    positions = torch.randint(0, 3000, (B,), device=dev, dtype=torch.int64)
    slot_mapping = torch.randperm(NB * BS, device=dev)[:B].to(torch.int64); slot_mapping[-3:] = -1  # 3 padded tokens
    q_scale = torch.tensor([0.37], device=dev, dtype=torch.float32); k_scale = torch.tensor([0.61], device=dev, dtype=torch.float32)
    # ---- reference ----
    q_pe_r, k_pe_r = rope(positions, q_pe.clone(), k_pe.clone())
    cache_store = torch.zeros(NB, BS, L + P, device=dev, dtype=torch.uint8)  # vLLM allocates fp8 caches as uint8 storage
    ops.concat_and_cache_mla(kv_c, k_pe_r.squeeze(1), cache_store, slot_mapping, "fp8", k_scale)
    cache_ref = cache_store.view(fp8)
    FMAX = torch.finfo(fp8).max
    q_ref = (torch.cat([ql_nope, q_pe_r], dim=-1).float() / q_scale).clamp(-FMAX, FMAX).to(fp8)
    # ---- fused ----
    cache_f = torch.zeros_like(cache_store).view(fp8)
    q_out = torch.empty(B, N, L + P, device=dev, dtype=fp8)
    cos, sin = rope.cos_sin_cache.chunk(2, dim=-1); cos = cos.contiguous(); sin = sin.contiguous()
    rocm_aiter_ops.fused_qk_rope_concat_and_cache_mla(ql_nope, q_pe, kv_c, k_pe.squeeze(1), cache_f.view(-1, 1, L + P), q_out,
                                                      slot_mapping, k_scale, q_scale, positions, cos, sin, is_neox=rope.is_neox_style)
    torch.cuda.synchronize()
    def cmp(name, a, b, exact=False):
        a = a.float(); b = b.float()
        d = (a - b).abs(); rel = d / b.abs().clamp(min=1e-2)
        if exact:
            assert d.max() == 0, f"{name}: max|d|={d.max()}"
        else:
            # rope runs in fp32 in the kernel vs bf16 in the reference: allow 1 fp8 ulp
            # (e4m3: 2^-3 relative) on a small fraction of elements
            assert (rel > 0.13).float().mean() < 0.01, f"{name}: too many mismatches"
            assert d.mean() < 0.02, f"{name}: mean|d|={d.mean()}"
    valid = slot_mapping >= 0
    cmp("q_out nope", q_out[valid][..., :L], q_ref[valid][..., :L], exact=True)
    cmp("q_out pe", q_out[valid][..., L:], q_ref[valid][..., L:])
    cache_mask = torch.zeros(NB * BS, dtype=torch.bool, device=dev); cache_mask[slot_mapping[valid]] = True
    cf = cache_f.view(NB * BS, L + P)[cache_mask]; cr = cache_ref.view(NB * BS, L + P)[cache_mask]
    cmp("cache nope", cf[:, :L], cr[:, :L], exact=True)
    cmp("cache pe", cf[:, L:], cr[:, L:])
    assert torch.equal(cache_f.view(NB * BS, -1)[~cache_mask].view(torch.uint8), cache_ref.view(NB * BS, -1)[~cache_mask].view(torch.uint8))
    assert torch.equal(q_pe, q_full[..., 128:])
