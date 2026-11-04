"""Host → device transfer of the field batches with pinned, asynchronous copies (training.gpu_prefetch, docs/38 §11).

A torch sparse tensor cannot be pinned, so a batch with the block-diagonal CSR operators (M, K, C, Ct, G, Gt, Kp) goes to
the device by pageable, effectively synchronous copies on the training thread.  Here the collate (worker side,
`field_dataset.field_collate_parts`) replaces every CSR tensor by its three dense parts (`SparseParts`, a NamedTuple:
DataLoader(pin_memory=True) pins it field by field, in its own thread, which also moves the unpickling of the batch off the
training thread).  `to_device` copies the parts with non_blocking=True (on a side CUDA stream when one is given) and
rebuilds the CSR tensors on the device; the training / validation / test step calls `wait_ready` first, which makes the
compute stream wait for the copy event (and tells the allocator that the compute stream uses the copied memory).  Lightning
issues the copy of batch N+1 right after the step N has been queued, so with a step that has no host synchronisation the
copy overlaps the GPU work of step N.

The batch is numerically the same as the plain one: the parts are the CSR tensor's own crow_indices / col_indices /
values, and `torch.sparse_csr_tensor(..., check_invariants=False)` is how `sparse.block_diag` builds them
(tests/test_throughput.py: exact equality on the CPU).  The CUDA branches (side stream, event, record_stream) cannot run
on a machine without a GPU and are not covered by the tests.
"""
from typing import NamedTuple, Tuple

import torch


class SparseParts(NamedTuple):
    """A CSR tensor as dense parts (pinnable, picklable through shared memory)."""
    crow: torch.Tensor
    col: torch.Tensor
    val: torch.Tensor
    size: Tuple[int, ...]


class DeviceBatch(dict):
    """A batch dict on the device whose copy was issued on a side stream: `ready_event` (recorded after the copy) and
    `keep` (the device tensors, for record_stream) until wait_ready has run."""
    ready_event = None
    keep = None


def _is_csr(x):
    return isinstance(x, torch.Tensor) and x.layout == torch.sparse_csr


def split_sparse(obj):
    """obj with every CSR tensor replaced by SparseParts (dicts, lists, tuples are walked; the parts are the tensor's own
    arrays, nothing is copied)."""
    if _is_csr(obj):
        return SparseParts(obj.crow_indices(), obj.col_indices(), obj.values(), tuple(obj.shape))
    if isinstance(obj, dict):
        return type(obj)((k, split_sparse(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)) and not isinstance(obj, SparseParts):
        return type(obj)(split_sparse(v) for v in obj)
    return obj


def _leaves(t):
    return [t.crow_indices(), t.col_indices(), t.values()] if _is_csr(t) else [t]


def join_sparse(obj, device, non_blocking=False, keep=None):
    """obj on `device`: tensors copied, SparseParts rebuilt as CSR tensors there.  keep (list or None) collects the
    device tensors."""
    if isinstance(obj, SparseParts):
        parts = [t.to(device, non_blocking=non_blocking) for t in (obj.crow, obj.col, obj.val)]
        if keep is not None:
            keep.extend(parts)
        return torch.sparse_csr_tensor(*parts, size=tuple(obj.size), check_invariants=False)
    if isinstance(obj, torch.Tensor):
        t = obj.to(device, non_blocking=non_blocking)
        if keep is not None:
            keep.extend(_leaves(t))
        return t
    if isinstance(obj, dict):
        return type(obj)((k, join_sparse(v, device, non_blocking, keep)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(join_sparse(v, device, non_blocking, keep) for v in obj)
    return obj


def to_device(batch, device, stream=None):
    """The batch on `device`.  CPU or stream None: plain (non_blocking on CUDA) copies on the current stream, CSR tensors
    rebuilt.  stream = a torch.cuda.Stream: the copies run on it and a DeviceBatch carries the event that the consuming
    step waits for (wait_ready)."""
    device = torch.device(device)
    if stream is None or device.type != 'cuda' or not isinstance(batch, dict):
        return join_sparse(batch, device, non_blocking=device.type == 'cuda')
    keep = []
    with torch.cuda.stream(stream):
        out = join_sparse(batch, device, True, keep)
        event = torch.cuda.Event()
        event.record(stream)
    out = DeviceBatch(out)
    out.ready_event, out.keep = event, keep
    return out


def wait_ready(batch):
    """Make the current CUDA stream wait for the copy of a DeviceBatch (no-op for any other batch); returns the batch."""
    event = getattr(batch, 'ready_event', None)
    if event is None:
        return batch
    stream = torch.cuda.current_stream()
    stream.wait_event(event)
    for t in batch.keep:             # allocated on the copy stream, used on this one: no reuse before this stream is done
        t.record_stream(stream)
    batch.ready_event = batch.keep = None
    return batch
