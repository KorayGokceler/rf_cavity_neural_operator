"""Mode debugging: tek batch forward+backward → her modun gradyanını ve tahminini kontrol et."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.models.gnot import GNOTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Dataset
ds = GNOTDataset("data/gnot_dataset.pkl", split="train")
loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=gnot_collate_fn)
batch = next(iter(loader))
batch = {k: v.to(device) for k, v in batch.items()}

# Model
model = GNOTModel(val_dim=6, grid_dim=2, theta_dim=1, embed_dim=256,
                  n_layers=6, n_heads=4, num_experts=4, num_field_modes=3).to(device)

# ═══════════════════════ TEST 1: Forward ═══════════════════════
print("\n" + "="*60)
print("TEST 1: FORWARD PASS — her modun tahmini")
print("="*60)

model.eval()
with torch.no_grad():
    outputs = model(batch)

pred = outputs['field']  # [B, N, 1]
true = batch['Y_field']
theta = batch['Theta_in'].squeeze(-1)
mask = batch['Mask']

for mode_val in [0, 1, 2]:
    idx = (theta == mode_val)
    n = idx.sum().item()
    if n == 0:
        print(f"  Mode {mode_val}: batch'te YOK!")
        continue
    p = pred[idx]
    t = true[idx]
    m = mask[idx]

    # Valid node'lardaki tahminler
    p_valid = p[m.unsqueeze(-1).expand_as(p)]
    t_valid = t[m.unsqueeze(-1).expand_as(t)]

    print(f"\n  Mode {mode_val} ({n} samples):")
    print(f"    Pred — mean: {p_valid.mean():.6f}, std: {p_valid.std():.6f}, "
          f"min: {p_valid.min():.6f}, max: {p_valid.max():.6f}")
    print(f"    True — mean: {t_valid.mean():.6f}, std: {t_valid.std():.6f}, "
          f"min: {t_valid.min():.6f}, max: {t_valid.max():.6f}")
    rel_l2 = torch.norm(p_valid - t_valid) / (torch.norm(t_valid) + 1e-8)
    print(f"    Rel L2: {rel_l2:.4f}")

# ═══════════════════════ TEST 2: Backward ═══════════════════════
print("\n" + "="*60)
print("TEST 2: BACKWARD — her modun gradient normu")
print("="*60)

model.train()
model.zero_grad()
outputs = model(batch)
pred = outputs['field']
true = batch['Y_field']
mask = batch['Mask']

# Tek tek mod loss'ları hesapla
for mode_val in [0, 1, 2]:
    model.zero_grad()
    outputs = model(batch)
    pred = outputs['field']

    idx = (theta == mode_val)
    if not idx.any():
        print(f"  Mode {mode_val}: batch'te YOK!")
        continue

    p = pred[idx]
    t = true[idx]
    m = mask[idx].unsqueeze(-1).float()
    loss = ((p - t) ** 2 * m).sum() / m.sum().clamp(min=1)
    loss.backward()

    print(f"\n  Mode {mode_val} — Loss: {loss.item():.6f}")

    # Shared trunk gradients
    trunk_grad = 0.0
    trunk_count = 0
    for name, param in model.shared_blocks.named_parameters():
        if param.grad is not None:
            trunk_grad += param.grad.norm().item()
            trunk_count += 1
    print(f"    Shared trunk grad norm: {trunk_grad:.6f} ({trunk_count} params)")

    # Mode-specific block gradients
    block_grad = 0.0
    block_count = 0
    for name, param in model.mode_field_blocks[mode_val].named_parameters():
        if param.grad is not None:
            block_grad += param.grad.norm().item()
            block_count += 1
        else:
            print(f"    ⚠️  mode_field_block[{mode_val}].{name}: grad=NONE!")
    print(f"    Mode block[{mode_val}] grad norm: {block_grad:.6f} ({block_count} params)")

    # Mode-specific head gradients
    head_grad = 0.0
    head_count = 0
    for name, param in model.field_heads[mode_val].named_parameters():
        if param.grad is not None:
            head_grad += param.grad.norm().item()
            head_count += 1
        else:
            print(f"    ⚠️  field_head[{mode_val}].{name}: grad=NONE!")
    print(f"    Mode head[{mode_val}] grad norm: {head_grad:.6f} ({head_count} params)")

    # Diğer modların block'ları gradient almıyor olmalı
    for other in [0, 1, 2]:
        if other == mode_val:
            continue
        other_grad = sum(p.grad.norm().item() for p in model.mode_field_blocks[other].parameters() if p.grad is not None)
        other_head_grad = sum(p.grad.norm().item() for p in model.field_heads[other].parameters() if p.grad is not None)
        if other_grad > 0 or other_head_grad > 0:
            print(f"    ⚠️  Mode {other} block/head gradient sızıntısı! block={other_grad:.6f}, head={other_head_grad:.6f}")
        else:
            print(f"    ✓  Mode {other} block/head gradient = 0 (izole)")

# ═══════════════════════ TEST 3: Sorting Check ═══════════════════════
print("\n" + "="*60)
print("TEST 3: SORT-UNSORT DOĞRULUK KONTROLÜ")
print("="*60)

sort_idx = theta.argsort()
unsort_idx = sort_idx.argsort()
theta_sorted = theta[sort_idx]
theta_recovered = theta_sorted[unsort_idx]
match = (theta == theta_recovered).all().item()
print(f"  Sort → unsort recovery: {'✓ DOĞRU' if match else '✗ YANLIŞ!'}")
print(f"  Sorted theta: {theta_sorted[:10].tolist()} ...")
print(f"  Mode counts: {[(theta == m).sum().item() for m in [0,1,2]]}")

print("\n" + "="*60)
print("DEBUG TAMAMLANDI")
print("="*60)
