"""Launcher helpers (scripts/serve_web.py): checkpoint auto-discovery and the frontend rebuild stamp."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import serve_web  # noqa: E402


def _touch(p, text='x', t=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    if t is not None:
        os.utime(p, (t, t))


def _ckpt(p, model_type='field3d', t=None):
    import torch
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'hyper_parameters': {'model_type': model_type}}, str(p))
    if t is not None:
        os.utime(p, (t, t))


def test_find_checkpoint_returns_the_newest_field3d_run(tmp_path):
    logs = tmp_path / 'training_logs'
    now = time.time()
    _ckpt(logs / 'old' / 'checkpoints' / 'last.ckpt', t=now - 100)
    _ckpt(logs / 'new' / 'version_0' / 'checkpoints' / 'epoch=3-val_rel_l2=0.1.ckpt', t=now)
    assert serve_web.find_checkpoint([str(logs)]) == str(logs / 'new')
    _ckpt(logs / 'loose.ckpt', t=now + 100)                    # a file directly in the root: the file itself
    assert serve_web.find_checkpoint([str(logs)]) == str(logs / 'loose.ckpt')
    _ckpt(logs / 'n0' / 'last.ckpt', 'eigenspace3d', t=now + 200)       # the removed N0 model: skipped
    _touch(logs / 'broken' / 'last.ckpt', 'not a checkpoint', t=now + 300)   # unreadable: skipped
    assert serve_web.find_checkpoint([str(logs)]) == str(logs / 'loose.ckpt')
    assert serve_web.find_checkpoint([str(tmp_path / 'missing')]) is None


def test_web_sources_hash_tracks_sources_only(tmp_path):
    web = tmp_path / 'web'
    _touch(web / 'src' / 'App.tsx', 'a')
    h0 = serve_web.web_sources_hash(str(web))
    for build_output in ('node_modules/x/index.js', 'dist/index.html', 'tsconfig.tsbuildinfo'):
        _touch(web / build_output)
    assert serve_web.web_sources_hash(str(web)) == h0
    _touch(web / 'src' / 'App.tsx', 'b')
    assert serve_web.web_sources_hash(str(web)) != h0
