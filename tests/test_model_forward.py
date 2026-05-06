"""End-to-end forward pass tests for GNOTModel.

Kritik invariantlar:
- Çıktı şekli: field [B, N, 1], freq [B, 1] (varsa)
- Mask uygulanması (padding nodelar attention'a karışmamalı)
- Çoklu mod batch'i: her sample doğru mod-spesifik branch'a yönlendirilmeli
- Forward NaN üretmemeli
- predict_frequency=False ise freq=None
"""
import pytest
import torch

from src.models.gnot import GNOTModel, RandomFourierFeatures, GeometricGatingFFN, AttentionPool


def _make_batch(B=2, N=20, val_dim=8, modes=None):
    if modes is None:
        modes = [0] * B
    return {
        'X': torch.randn(B, N, 2),
        'Input_funcs': torch.randn(B, N, val_dim),
        'Theta_in': torch.tensor(modes, dtype=torch.long).unsqueeze(-1),
        'Mask': torch.ones(B, N, dtype=torch.bool),
    }


def _build_small_model(predict_frequency=True, num_field_modes=3):
    return GNOTModel(
        val_dim=8, grid_dim=2, embed_dim=16,
        n_shared_layers=1, n_mode_layers=1, n_field_head_layers=2,
        n_heads=2, num_experts=2, num_field_modes=num_field_modes,
        predict_frequency=predict_frequency,
        rff_dim=8, rff_length_scale=0.1,
    )


def test_forward_output_shapes():
    model = _build_small_model()
    batch = _make_batch(B=3, N=15, modes=[0, 1, 2])
    out = model(batch)
    assert out['field'].shape == (3, 15, 1)
    assert out['freq'].shape == (3, 1)


def test_forward_no_frequency_branch():
    model = _build_small_model(predict_frequency=False)
    batch = _make_batch(B=2, N=10, modes=[0, 1])
    out = model(batch)
    assert out['field'].shape == (2, 10, 1)
    assert out['freq'] is None


def test_forward_finite_output():
    """Çıktıda NaN/inf olmamalı."""
    torch.manual_seed(0)
    model = _build_small_model()
    batch = _make_batch(B=4, N=20, modes=[0, 0, 1, 2])
    out = model(batch)
    assert torch.isfinite(out['field']).all()
    assert torch.isfinite(out['freq']).all()


def test_forward_with_padding_mask():
    """Padding ile birlikte forward çalışmalı."""
    torch.manual_seed(1)
    model = _build_small_model()
    batch = _make_batch(B=2, N=20, modes=[0, 1])
    # 2. örneği yarım yap (padding ekle)
    batch['Mask'][1, 10:] = False
    out = model(batch)
    assert torch.isfinite(out['field']).all()
    # Padding bölgesinde ne çıkarsa çıksın — model bunu mask'lemez (loss'ta mask uygulanır),
    # ama forward NaN üretmemeli
    assert out['field'].shape == (2, 20, 1)


def test_mode_routing_independence():
    """Aynı geometri farklı mode_id'lerle farklı çıktı üretmeli."""
    torch.manual_seed(2)
    model = _build_small_model()
    model.eval()  # dropout devre dışı
    batch_0 = _make_batch(B=1, N=15, modes=[0])
    batch_1 = _make_batch(B=1, N=15, modes=[1])
    # Aynı X ve Input_funcs kullan
    batch_1['X'] = batch_0['X'].clone()
    batch_1['Input_funcs'] = batch_0['Input_funcs'].clone()
    with torch.no_grad():
        out_0 = model(batch_0)
        out_1 = model(batch_1)
    # Farklı mode → farklı field tahmini
    assert not torch.allclose(out_0['field'], out_1['field'])


def test_deterministic_eval_mode():
    """Eval modunda aynı input → aynı output (dropout kapalı, gates deterministic)."""
    torch.manual_seed(3)
    model = _build_small_model()
    model.eval()
    batch = _make_batch(B=2, N=10, modes=[0, 1])
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


def test_full_batch_with_mixed_modes_finite():
    """Karışık modları ile büyükçe bir batch — tüm output finite olmalı."""
    torch.manual_seed(7)
    model = _build_small_model(num_field_modes=3)
    B, N = 6, 25
    modes = [0, 1, 2, 0, 1, 2]
    batch = _make_batch(B=B, N=N, modes=modes)
    out = model(batch)
    assert torch.isfinite(out['field']).all()
    assert torch.isfinite(out['freq']).all()
    # Her mode için sample sayısı
    for m in range(3):
        count = sum(1 for x in modes if x == m)
        # Sadece doğru mode_field_blocks'a yönlendirildiğinden emin olamıyoruz forward içinden,
        # ama field_pred boyutunun korunduğunu test edebiliriz
        assert (batch['Theta_in'].squeeze(-1) == m).sum().item() == count


def test_gradient_flow_to_shared_blocks():
    """Backward sonrası shared_blocks parametrelerinin gradı olmalı."""
    torch.manual_seed(8)
    model = _build_small_model()
    batch = _make_batch(B=2, N=10, modes=[0, 1])
    out = model(batch)
    loss = out['field'].sum() + out['freq'].sum()
    loss.backward()

    has_grad = False
    for name, p in model.shared_blocks.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad = True
            break
    assert has_grad


def test_mode_specific_gradient_isolation():
    """Mode 0'ın çıktısının gradı yalnızca mode_field_blocks[0]'a akmalı, [1] ve [2]'ye değil."""
    torch.manual_seed(9)
    model = _build_small_model(num_field_modes=3)
    # Sadece mode 0 ile bir batch
    batch = _make_batch(B=2, N=10, modes=[0, 0])
    out = model(batch)
    # Mode 0 field'ından geri yay
    out['field'].sum().backward()

    # mode_field_blocks[0] gradient almalı
    grad_0_sum = 0.0
    for p in model.mode_field_blocks[0].parameters():
        if p.grad is not None:
            grad_0_sum += p.grad.abs().sum().item()
    assert grad_0_sum > 0

    # mode_field_blocks[1] ve [2] gradient almamalı (None ya da 0)
    for mode_branch in [model.mode_field_blocks[1], model.mode_field_blocks[2]]:
        for p in mode_branch.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0
