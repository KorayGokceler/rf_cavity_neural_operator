"""End-to-end forward pass tests for GNOTModel (spectral-subspace arch).

Invariants:
- Output shape: field [B, N, K], freq [B, K]
- freq is always sorted ascending
- No mode_idx / Theta_in input — every slot runs for every sample
- Mask handling (padding nodes do not produce NaN)
- Forward must not produce NaN
"""
from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.models.gnot import (GNOTModel, GNOT, GeometricGatingFFN, AttentionPool,
                             LinearAttention, _apply_gram_schmidt)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_batch(B=2, N=20, val_dim=8):
    return {
        'X': torch.randn(B, N, 2),
        'Input_funcs': torch.randn(B, N, val_dim),
        'Mask': torch.ones(B, N, dtype=torch.bool),
    }


def _build_small_model(predict_frequency=True, num_field_modes=3, **kw):
    return GNOTModel(
        val_dim=8, grid_dim=2, embed_dim=16,
        n_shared_layers=1, n_mode_layers=1, n_field_head_layers=2,
        n_heads=2, num_experts=2, num_field_modes=num_field_modes,
        predict_frequency=predict_frequency,
        rff_dim=8, rff_length_scale=0.1, **kw,
    )


def test_forward_output_shapes():
    model = _build_small_model(num_field_modes=3)
    batch = _make_batch(B=3, N=15)
    out = model(batch)
    assert out['field'].shape == (3, 15, 3)
    assert out['freq'].shape == (3, 3)


def test_freq_always_sorted_ascending():
    """The global frequency head output is always sorted ascending."""
    torch.manual_seed(11)
    model = _build_small_model(num_field_modes=3)
    batch = _make_batch(B=5, N=20)
    out = model(batch)
    f = out['freq']
    assert (f[:, 1:] >= f[:, :-1]).all()


def test_alias_GNOT_equals_model():
    assert GNOT is GNOTModel


def test_forward_finite_output():
    """No NaN/inf in outputs."""
    torch.manual_seed(0)
    model = _build_small_model(num_field_modes=3)
    batch = _make_batch(B=4, N=20)
    out = model(batch)
    assert torch.isfinite(out['field']).all()
    assert torch.isfinite(out['freq']).all()


def test_forward_with_padding_mask():
    """Forward must work (and stay finite) with padding."""
    torch.manual_seed(1)
    model = _build_small_model(num_field_modes=3)
    batch = _make_batch(B=2, N=20)
    batch['Mask'][1, 10:] = False
    out = model(batch)
    assert torch.isfinite(out['field']).all()
    assert out['field'].shape == (2, 20, 3)
    assert out['freq'].shape == (2, 3)


def test_all_slots_distinct():
    """Each of the K slots should generally produce a different field."""
    torch.manual_seed(2)
    model = _build_small_model(num_field_modes=3)
    model.eval()
    batch = _make_batch(B=1, N=15)
    with torch.no_grad():
        out = model(batch)
    f = out['field'][0]  # [N, K]
    assert not torch.allclose(f[:, 0], f[:, 1])
    assert not torch.allclose(f[:, 1], f[:, 2])


def test_deterministic_eval_mode():
    """Eval mode: same input → same output."""
    torch.manual_seed(3)
    model = _build_small_model(num_field_modes=3)
    model.eval()
    batch = _make_batch(B=2, N=10)
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)
    assert torch.allclose(out1['field'], out2['field'])
    assert torch.allclose(out1['freq'], out2['freq'])


def test_geometric_gating_ffn_weighted_sum():
    """Gate ağırlıkları toplamı 1 ise final_output, expert outputlarının ağırlıklı toplamı olmalı."""
    torch.manual_seed(4)
    ffn = GeometricGatingFFN(embed_dim=8, num_experts=2, dropout=0.0)
    x = torch.randn(2, 5, 8)
    # Tek expert'a tam ağırlık ver: gate = [1, 0]
    gw = torch.zeros(2, 5, 2)
    gw[..., 0] = 1.0
    out = ffn(x, gw)
    expected = ffn.experts[0](x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_geometric_gating_ffn_balanced():
    """50/50 gate → 0.5*E0 + 0.5*E1."""
    torch.manual_seed(5)
    ffn = GeometricGatingFFN(embed_dim=4, num_experts=2, dropout=0.0)
    x = torch.randn(1, 3, 4)
    gw = torch.full((1, 3, 2), 0.5)
    out = ffn(x, gw)
    expected = 0.5 * ffn.experts[0](x) + 0.5 * ffn.experts[1](x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_attention_pool_output_shape():
    pool = AttentionPool(embed_dim=8, num_heads=2)
    x = torch.randn(3, 12, 8)
    mask = torch.ones(3, 12, dtype=torch.bool)
    mask[2, 6:] = False  # 3. örneği yarım yap
    out = pool(x, mask)
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()


def test_attention_pool_no_mask():
    pool = AttentionPool(embed_dim=8, num_heads=2)
    x = torch.randn(2, 10, 8)
    out = pool(x, mask=None)
    assert out.shape == (2, 8)


def test_full_batch_finite():
    """A larger batch — all outputs finite, correct shapes."""
    torch.manual_seed(7)
    model = _build_small_model(num_field_modes=3)
    B, N = 6, 25
    batch = _make_batch(B=B, N=N)
    out = model(batch)
    assert torch.isfinite(out['field']).all()
    assert torch.isfinite(out['freq']).all()
    assert out['field'].shape == (B, N, 3)
    assert out['freq'].shape == (B, 3)


def test_gradient_flow_to_mode_field_blocks():
    """After backward, all mode_field_blocks must receive gradient."""
    torch.manual_seed(8)
    model = _build_small_model(num_field_modes=3)
    batch = _make_batch(B=2, N=10)
    out = model(batch)
    loss = out['field'].sum() + out['freq'].sum()
    loss.backward()

    for k, blocks in enumerate(model.mode_field_blocks):
        has_grad = False
        for name, p in blocks.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad = True
                break
        assert has_grad, f"mode_field_blocks[{k}] has no gradient"


def test_all_slot_decoders_receive_gradient():
    """Every slot decoder runs for every sample now, so all mode_field_blocks
    must receive gradient from the combined field output."""
    torch.manual_seed(9)
    model = _build_small_model(num_field_modes=3)
    batch = _make_batch(B=2, N=10)
    out = model(batch)
    out['field'].sum().backward()

    for k in range(3):
        gsum = 0.0
        for p in model.mode_field_blocks[k].parameters():
            if p.grad is not None:
                gsum += p.grad.abs().sum().item()
        assert gsum > 0, f"slot {k} decoder received no gradient"


# ─── Masking / symmetry ──────────────────────────────────────────────────────

def _pad(batch, n_extra):
    B = batch['X'].shape[0]
    return {
        'X': torch.cat([batch['X'], torch.randn(B, n_extra, 2)], dim=1),
        'Input_funcs': torch.cat([batch['Input_funcs'],
                                  torch.randn(B, n_extra, batch['Input_funcs'].shape[-1])], dim=1),
        'Mask': torch.cat([batch['Mask'], torch.zeros(B, n_extra, dtype=torch.bool)], dim=1),
    }


@pytest.mark.parametrize("orthonormalize", [False, True])
def test_padding_invariance(orthonormalize):
    """Appending masked garbage nodes must not change any valid-node output
    (attention keys, pooling and Gram-Schmidt must all respect the mask)."""
    torch.manual_seed(12)
    model = _build_small_model(orthonormalize_output=orthonormalize).eval()
    batch = _make_batch(B=2, N=15)
    with torch.no_grad():
        o1 = model(batch)
        o2 = model(_pad(batch, 6))
    assert torch.allclose(o1['field'], o2['field'][:, :15], atol=1e-5)
    assert torch.allclose(o1['freq'], o2['freq'], atol=1e-5)
    assert o2['field'][:, 15:].abs().max() == 0


def test_permutation_equivariance():
    torch.manual_seed(13)
    model = _build_small_model().eval()
    batch = _make_batch(B=2, N=15)
    perm = torch.randperm(15)
    with torch.no_grad():
        o1 = model(batch)
        o2 = model({k: v[:, perm] for k, v in batch.items()})
    assert torch.allclose(o1['field'][:, perm], o2['field'], atol=1e-5)
    assert torch.allclose(o1['freq'], o2['freq'], atol=1e-5)


# ─── Gram-Schmidt output option ──────────────────────────────────────────────

def test_gram_schmidt_backward():
    """The former in-place column writes made backward raise
    'modified by an inplace operation' -> orthonormalize_output was untrainable."""
    torch.manual_seed(14)
    field = torch.randn(2, 10, 3, requires_grad=True)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 7:] = False
    q = _apply_gram_schmidt(field, mask)
    q.pow(3).sum().backward()
    assert torch.isfinite(field.grad).all()
    G = torch.einsum('bnk,bnl->bkl', q, q)
    assert torch.allclose(G, torch.eye(3).expand_as(G), atol=1e-5)
    assert q[1, 7:].abs().max() == 0


def test_orthonormalize_output_trainable_and_peak_normalized():
    """Unit-L2 columns are ~1/sqrt(N) per node while targets are max|Y| = 1
    normalized, so the field is rescaled to a unit peak (orthogonality kept)."""
    torch.manual_seed(15)
    model = _build_small_model(orthonormalize_output=True).train()
    batch = _make_batch(B=2, N=30)
    batch['Mask'][1, 20:] = False
    out = model(batch)
    (out['field'].pow(2).sum() + out['freq'].sum()).backward()
    f = out['field'].detach()
    assert torch.allclose(f.abs().amax(dim=1), torch.ones(2, 3), atol=1e-5)
    G = torch.einsum('bnk,bnl->bkl', f, f)
    off = G - torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))
    assert off.abs().max() < 1e-4


# ─── Checkpointing / dropout / config ────────────────────────────────────────

def test_gradient_checkpointing_matches_plain():
    batch = _make_batch(B=2, N=12)
    results = []
    for ckpt in (False, True):
        torch.manual_seed(16)
        model = _build_small_model(use_checkpoint=ckpt).train()
        out = model(batch)
        (out['field'].pow(2).sum() + out['freq'].sum()).backward()
        results.append((out['field'].detach(),
                        torch.cat([p.grad.flatten() for p in model.parameters()
                                   if p.grad is not None])))
    assert torch.allclose(results[0][0], results[1][0], atol=1e-6)
    assert torch.allclose(results[0][1], results[1][1], atol=1e-6)


def test_linear_attention_dropout_is_applied():
    """`dropout` used to be accepted and silently ignored by LinearAttention."""
    torch.manual_seed(17)
    attn = LinearAttention(embed_dim=8, num_heads=2, dropout=0.5)
    x = torch.randn(2, 6, 8)
    attn.train()
    assert not torch.allclose(attn(x, x, x), attn(x, x, x))
    attn.eval()
    assert torch.equal(attn(x, x, x), attn(x, x, x))


def test_val_dim_mismatch_raises_clear_error():
    model = _build_small_model()
    with pytest.raises(ValueError, match="val_dim=8"):
        model(_make_batch(val_dim=12))


def test_default_config_forward_backward_cpu():
    """configs/default.yaml, scaled down (embed_dim/rff_dim), train-mode
    forward + backward on CPU with padding."""
    mc = load_config(str(REPO_ROOT / 'configs' / 'default.yaml')).model
    torch.manual_seed(18)
    model = GNOTModel(
        val_dim=mc.val_dim, grid_dim=mc.grid_dim, embed_dim=16,
        n_shared_layers=mc.n_shared_layers, n_mode_layers=mc.n_mode_layers,
        n_field_head_layers=mc.n_field_head_layers, n_heads=mc.n_heads // 4,
        num_experts=mc.num_experts, num_field_modes=mc.num_field_modes,
        use_checkpoint=True, predict_frequency=mc.predict_frequency,
        dropout=0.1, rff_dim=8, rff_length_scale=mc.rff_length_scale,
        n_basis=mc.n_basis, orthonormalize_output=mc.orthonormalize_output,
    ).train()
    batch = _make_batch(B=2, N=24, val_dim=mc.val_dim)
    batch['Mask'][0, 18:] = False
    out = model(batch)
    assert out['field'].shape == (2, 24, mc.num_field_modes)
    (out['field'].pow(2).mean() + out['freq'].pow(2).mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
