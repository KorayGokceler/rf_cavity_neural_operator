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


def test_find_checkpoint_returns_the_newest_training_run(tmp_path):
    logs = tmp_path / 'training_logs'
    now = time.time()
    _touch(logs / 'old' / 'checkpoints' / 'last.ckpt', t=now - 100)
    _touch(logs / 'new' / 'version_0' / 'checkpoints' / 'epoch=3-val_rel_l2=0.1.ckpt', t=now)
    assert serve_web.find_checkpoint([str(logs)]) == str(logs / 'new')
    _touch(logs / 'loose.ckpt', t=now + 100)                    # a file directly in the root: the file itself
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
