"""Scheduler of the dataset generator (run_samples + main): results are written as they finish, a worker that dies
or hangs fails only its own sample.  A fake sample function runs in a subprocess (the worker processes fork; the test
process has usually run NGSolve threads, see conftest.run_generator)."""
import subprocess
import sys
import textwrap
import time

import h5py
import pytest

from tests.conftest import REPO_ROOT

DRIVER = textwrap.dedent('''
    import os, sys, time, signal
    sys.path.insert(0, {repo!r})
    import numpy as np
    import src.data_gen.dataset_generator_3d as gen

    def fake(s_id):
        if s_id == {bad}:
            {action}
        time.sleep({delay} * (s_id % 2) if {long_id} < 0 else (6.0 if s_id == {long_id} else 0.3))
        return {{"id": s_id, "t_mesh": 0.0, "t_solve": 0.0, "size": (1, 1)}}

    def fake_write(f, res):
        g = f.create_group("sample_%04d" % res["id"])
        g.attrs["id"] = res["id"]
        return g

    gen.generate_sample_data = fake
    gen.write_sample = fake_write
    t0 = time.time()
    gen.main(sys.argv[1:])
    print('MAIN_WALL', time.time() - t0)
''')


def _run(tmp_path, bad, action, n=6, timeout="3600", delay=0.0, n_workers="2", long_id=-1):
    script = tmp_path / "drv.py"
    script.write_text(DRIVER.format(repo=str(REPO_ROOT), bad=bad, action=action, delay=delay, long_id=long_id))
    out = tmp_path / "o.h5"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script), "--n_total", str(n), "--n_workers", n_workers,
                        "--families", "pillbox", "--sample_timeout", timeout, "--h5_filename", str(out)],
                       capture_output=True, text=True, timeout=120)
    return r, out, time.time() - t0


@pytest.mark.parametrize("action", ["os._exit(1)", "os.kill(os.getpid(), signal.SIGKILL)"])
def test_dead_worker_fails_only_its_sample(tmp_path, action):
    r, out, dt = _run(tmp_path, 2, action)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sample 2: worker died" in r.stdout and "skipped" in r.stdout
    assert "6/6 done (5 ok, 1 failed)" in r.stdout and "Done: 5 samples (1 failed)" in r.stdout
    with h5py.File(out, "r") as f:
        assert sorted(f.keys()) == [f"sample_{i:04d}" for i in (0, 1, 3, 4, 5)]
    assert dt < 30


def test_timeout_counts_from_sample_start_and_costs_once(tmp_path):
    r, out, dt = _run(tmp_path, 1, "time.sleep(600)", n=6, timeout="3")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Sample 1 timed out after 3s, skipped." in r.stdout
    with h5py.File(out, "r") as f:
        assert sorted(f.keys()) == [f"sample_{i:04d}" for i in (0, 2, 3, 4, 5)]
    assert dt < 20            # ~ timeout, not timeout x n


def test_results_are_written_before_the_run_ends(tmp_path, monkeypatch):
    """Sample 1 hangs; the finished samples must already be in the .partial while the run is still waiting."""
    script = tmp_path / "drv.py"
    script.write_text(DRIVER.format(repo=str(REPO_ROOT), bad=1, action="time.sleep(600)", delay=0.0, long_id=-1))
    out = tmp_path / "o.h5"
    monkeypatch.setenv("HDF5_USE_FILE_LOCKING", "FALSE")      # read the file the writer still holds open
    p = subprocess.Popen([sys.executable, str(script), "--n_total", "4", "--n_workers", "2", "--families", "pillbox",
                          "--h5_filename", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        keys = []
        for _ in range(100):
            time.sleep(0.2)
            try:
                with h5py.File(str(out) + ".partial", "r") as f:
                    keys = sorted(f.keys())
            except OSError:
                continue
            if len(keys) >= 3:
                break
        assert p.poll() is None                      # still waiting for the hung sample
        assert keys == ["sample_0000", "sample_0002", "sample_0003"]
    finally:
        p.kill()
        p.wait()


def test_pool_stays_full_no_chunk_barrier(tmp_path):
    """One 6 s sample + 30 x 0.3 s with 3 workers: the two free workers drain the short ones while the long one runs,
    wall ~ max(6, 30 * 0.3 / 2 = 4.5) s.  With chunks of 4 * n_workers = 12 ids and a barrier it would be >= 6 + 2 * 1.8."""
    r, out, _ = _run(tmp_path, -99, "pass", n=31, n_workers="3", long_id=0)
    assert r.returncode == 0, r.stdout + r.stderr
    wall = float([ln for ln in r.stdout.splitlines() if ln.startswith("MAIN_WALL")][0].split()[1])
    assert "Done: 31 samples (0 failed)" in r.stdout
    assert wall < 8.5, wall
