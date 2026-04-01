"""
Regression test for the GDN ratio=1 contiguity bug.

When num_v_heads == num_k_heads, fix_query_key_value_ordering produces
non-contiguous a/b tensors unless .contiguous() is applied after reshape.
The downstream fused_gdn_gating triton kernel assumes contiguous layout,
so non-contiguous a/b leads to reading wrong values.

This was never caught because the original Qwen3 GDN uses ratio=2
(num_v_heads = 2 * num_k_heads), where reshape forces a contiguous copy.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(42)


def fix_query_key_value_ordering(
    mixed_qkvz, mixed_ba, num_k_heads, num_v_heads, attn_tp_size, head_k_dim, head_v_dim
):
    """Matches Qwen3GatedDeltaNet.fix_query_key_value_ordering in qwen3_next.py."""
    new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
        num_k_heads // attn_tp_size,
        (
            head_k_dim
            + head_k_dim
            + (head_v_dim + head_v_dim) * num_v_heads // num_k_heads
        ),
    )
    new_tensor_shape_ba = mixed_ba.size()[:-1] + (
        num_k_heads // attn_tp_size,
        2 * num_v_heads // num_k_heads,
    )

    mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
    mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

    split_arg_list_qkvz = [
        head_k_dim,
        head_k_dim,
        (num_v_heads // num_k_heads * head_v_dim),
        (num_v_heads // num_k_heads * head_v_dim),
    ]
    split_arg_list_ba = [
        num_v_heads // num_k_heads,
        num_v_heads // num_k_heads,
    ]

    query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
    b, a = torch.split(mixed_ba, split_arg_list_ba, dim=2)

    value = value.reshape(value.size(0), -1, head_v_dim)
    z = z.reshape(z.size(0), -1, head_v_dim)
    b = b.reshape(b.size(0), num_v_heads // attn_tp_size).contiguous()
    a = a.reshape(a.size(0), num_v_heads // attn_tp_size).contiguous()
    query, key, value = map(lambda x: x.reshape(x.shape[0], -1), (query, key, value))
    mixed_qkv = torch.cat((query, key, value), dim=-1)

    return mixed_qkv, z, b, a


def kernel_reads_correct_values(tensor):
    """Check whether a triton kernel assuming contiguous layout reads correct values.

    The fused_gdn_gating kernel indexes with: off = i_b * NUM_HEADS + head_idx,
    which is equivalent to reading from a contiguous [batch, num_heads] tensor.
    """
    num_heads = tensor.shape[1]
    as_kernel_sees = tensor.as_strided(tensor.shape, (num_heads, 1))
    return torch.allclose(tensor.contiguous(), as_kernel_sees)


class TestGDNContiguity(CustomTestCase):
    def _run_ratio(self, num_k_heads, num_v_heads):
        head_k_dim = 128
        head_v_dim = 128
        attn_tp_size = 1
        seq_len = 64

        proj_qkvz = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim * 2
        proj_ba = num_v_heads * 2

        mixed_qkvz = torch.randn(seq_len, proj_qkvz)
        mixed_ba = torch.randn(seq_len, proj_ba)

        mixed_qkv, z, b, a = fix_query_key_value_ordering(
            mixed_qkvz,
            mixed_ba,
            num_k_heads,
            num_v_heads,
            attn_tp_size,
            head_k_dim,
            head_v_dim,
        )

        ratio = num_v_heads // num_k_heads
        self.assertTrue(
            a.is_contiguous(),
            f"ratio={ratio}: a not contiguous — "
            f"shape={list(a.shape)}, stride={list(a.stride())}",
        )
        self.assertTrue(
            b.is_contiguous(),
            f"ratio={ratio}: b not contiguous — "
            f"shape={list(b.shape)}, stride={list(b.stride())}",
        )
        self.assertTrue(
            kernel_reads_correct_values(a),
            f"ratio={ratio}: kernel reads wrong values for a",
        )
        self.assertTrue(
            kernel_reads_correct_values(b),
            f"ratio={ratio}: kernel reads wrong values for b",
        )

    def test_ratio_1(self):
        """num_v_heads == num_k_heads — the case that triggered the bug."""
        self._run_ratio(num_k_heads=16, num_v_heads=16)

    def test_ratio_2(self):
        """num_v_heads == 2 * num_k_heads — original Qwen3 GDN config."""
        self._run_ratio(num_k_heads=16, num_v_heads=32)

    def test_ratio_4(self):
        """num_v_heads == 4 * num_k_heads."""
        self._run_ratio(num_k_heads=16, num_v_heads=64)


if __name__ == "__main__":
    unittest.main()
