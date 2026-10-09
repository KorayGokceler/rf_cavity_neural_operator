"""Training from the web UI (src/service/training.py + /api/train/*): merge two datasets' H5 shards,
train a tiny model, follow progress.json, resume with more epochs, switch the served model."""
import os
import re
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient                                   # noqa: E402

from src.data_gen import dataset_generator_3d as gen                         # noqa: E402
from src.service import api as A                                            # noqa: E402
from src.service import training as T                                       # noqa: E402
from src.service.model import ModelService                                  # noqa: E402


def test_presets_match_the_cluster_config():
    """cluster/truba/config.sh model_overrides and the UI presets describe the same models."""
    sh = open(os.path.join(ROOT, 'cluster', 'truba', 'config.sh')).read()
    for name in ('small', 'base', 'large', 'xl'):
        line = re.search(rf'^\s*{name}\)\s*echo "([^"]+)"', sh, re.M).group(1)
        assert sorted(line.split()) == sorted(T.preset_overrides(name)), name


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    d = tmp_path_factory.mktemp("trainui")
    for fam, start in (("pillbox_pipes", 3 * 2 ** 19), ("reentrant", 1 * 2 ** 19)):
        (d / "data" / "T" / "h5" / fam).mkdir(parents=True)
        gen.main(["--n_total", "6", "--n_workers", "2", "--mesh_size", "0.3", "--n_eigen_modes", "6",
                  "--families", fam, "--start_id", str(start),
                  "--h5_filename", str(d / "data" / "T" / "h5" / fam / f"{fam}_s00000.h5")])
    return d


def _wait(c, name, until=("finished", "failed", "stopped"), timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/train/runs/{name}").json()
        if r["status"] in until:
            return r
        time.sleep(0.5)
    raise AssertionError(f"run {name} did not finish: {r}")


def test_train_resume_and_load(tree, monkeypatch):
    monkeypatch.setenv("RFCAV_DATA", str(tree / "data"))
    monkeypatch.setenv("RFCAV_RUNS_ROOT", str(tree / "runs"))
    monkeypatch.delenv("RFCAV_GEN_ROOT", raising=False)
    svc = ModelService(None, "cpu")
    c = TestClient(A.create_app(svc))
    cfg = c.get("/api/train/config").json()
    assert cfg["enabled"] and "tiny" in cfg["presets"] and not cfg["busy"]
    ids = [d["id"] for d in c.get("/api/datasets").json()]
    assert len(ids) == 2
    body = {"name": "ui1", "datasets": ids, "preset": "tiny", "epochs": 1, "batch_size": 2, "n_modes": 4}
    r = c.post("/api/train/start", json=body)
    assert r.status_code == 200, r.text
    assert c.post("/api/train/start", json=dict(body, name="ui2")).status_code == 409      # one at a time
    r = _wait(c, "ui1")
    assert r["status"] == "finished", r["log"][-15:]
    assert r["epoch"] == 1 and len(r["history"]) == 1 and "val/field_rel_l2" in r["history"][0]
    assert r["test"] and r["best"]["path"].endswith(".ckpt") and r["resumable"]
    assert r["params"]["sources"] and len(r["params"]["data"]["h5"]) == 2
    assert c.post("/api/train/start", json=body).status_code == 409                       # name taken

    r = c.post("/api/train/runs/ui1/resume", json={"epochs": 2})
    assert r.status_code == 200, r.text
    r = _wait(c, "ui1")
    assert r["status"] == "finished" and [h["epoch"] for h in r["history"]] == [0, 1]
    assert r["data_cached"]                                                                # merged PKL reused

    assert svc.untrained
    info = c.post("/api/model/load", json={"run": "ui1"}).json()
    assert info["untrained"] is False and info["run"] == "ui1" and info["n_modes"] == 4
    assert c.post("/api/model/load", json={"run": ".."}).status_code == 422
    assert [x["name"] for x in c.get("/api/train/runs").json()] == ["ui1"]

    # stop a long run after its first epoch: stopped, resumable from last.ckpt
    assert c.post("/api/train/start", json=dict(body, name="ui3", epochs=500)).status_code == 200
    t0 = time.time()
    while c.get("/api/train/runs/ui3").json()["epoch"] < 1 and time.time() - t0 < 300:
        time.sleep(0.3)
    assert c.post("/api/train/stop").json()["stopped"] == "ui3"
    r = _wait(c, "ui3", until=("stopped",))
    assert r["resumable"] and 1 <= r["epoch"] < 500
    assert not c.get("/api/train/config").json()["busy"]


def test_training_off_without_runs_root(monkeypatch):
    monkeypatch.delenv("RFCAV_RUNS_ROOT", raising=False)
    c = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    assert c.get("/api/train/config").json()["enabled"] is False
    assert c.post("/api/train/start", json={"name": "x", "datasets": ["d"]}).status_code == 503
