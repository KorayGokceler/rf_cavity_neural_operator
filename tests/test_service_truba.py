"""TRUBA page (src/service/truba.py + /api/truba/*): the one-line command, validation, defaults."""
import os
import shlex
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.service import truba as TR  # noqa: E402

FORM = TR.defaults(ROOT)["form"]


def test_default_field_pipeline_is_one_runnable_line():
    r = TR.build_command(FORM)
    cmd = r["command"]
    assert "\n" not in cmd and cmd.count("bash cluster/truba/run.sh all") == 1
    assert cmd.startswith("[ -d ~/rf_cavity_neural_operator ] || git clone ")
    assert "LABELS=field" in r["run_args"] and "N_PER_FAMILY=1024" in r["run_args"]
    assert "FAMILIES=" not in r["run_args"]                       # all train families = run.sh's default
    assert "freeform" not in r["families"]                        # field labels need a CAD solid
    assert r["tag"] == "F_p3p2_ms0.10_k10_mf0.05_v1"
    assert r["estimate"]["workers_per_job"] == 56 // 8
    shlex.split(cmd)                                              # well-formed shell words
    r = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_values_are_written_only_when_they_differ_from_config_sh():
    r = TR.build_command(dict(FORM, action="dataset", labels="n0", families=["elliptical", "hwr"], mesh_size=0.12,
                              test=True, n_per_family=""))
    assert r["run_args"] == "bash cluster/truba/run.sh dataset FAMILIES=elliptical,hwr MESH_SIZE=0.12 TEST=1"
    assert r["tag"] == "E_ms0.12_k10_v1_test"
    r = TR.build_command(dict(FORM, action="train", exp="f2", model="large", gpus=4, batch=1, min_fillet=0.08,
                              cache_dir="/arf/scratch/$USER/ops", train_extra=["training.patience=30"]))
    a = r["run_args"]
    for s in ("LABELS=field", "MIN_FILLET=0.08", "GPUS=4", "EXP=f2", "MODEL=large", "BATCH=1",
              'CACHE_DIR="/arf/scratch/$USER/ops"', "TRAIN_EXTRA=training.patience=30"):
        assert s in a, s
    assert "N_PER_FAMILY" not in a and "THREADS" not in a        # dataset-only settings stay out of `train`
    assert r["tag"] == "F_p3p2_ms0.10_k10_mf0.08_v1"
    assert TR.build_command(dict(FORM, action="all", families=TR.OOD_FAMILIES + ("elliptical",)))["group"] == \
        ",".join(("elliptical",) + TR.OOD_FAMILIES)
    assert TR.build_command(dict(FORM, dry_run=True))["run_args"].endswith(" --dry-run")


def test_adaptive_label_settings_and_shard_time_note():
    r = TR.build_command(FORM)
    assert "shard_time" not in r["notes"]                          # ~8.5 h per 1024-geometry shard < 12 h
    assert "ADAPT" not in r["run_args"] and "TOL_F" not in r["run_args"]
    r = TR.build_command(dict(FORM, action="dataset", adapt=False, tol_f=1e-5, model_max_elements=6000,
                              time="0-06:00:00"))
    for s in ("ADAPT=0", "TOL_F=1e-05", "MODEL_MAX_ELEMENTS=6000", "TIME=0-06:00:00"):
        assert s in r["run_args"], s
    assert "shard_time" not in r["notes"]                          # one solve: a third of the time
    assert "shard_time" in TR.build_command(dict(FORM, action="dataset", time="0-06:00:00"))["notes"]
    assert "shard_time" not in TR.build_command(dict(FORM, action="train", time="0-06:00:00"))["notes"]
    assert TR._hours("1-02:30:00") == 26.5 and TR._hours("12:00:00") == 12


@pytest.mark.parametrize("bad", [
    {"partition": "orfoz; rm -rf ~"}, {"tag": "a b"}, {"time": "12h"}, {"branch": "main && id"},
    {"repo": "https://x/y; id"}, {"train_extra": ["x=$(id)"]}, {"cache_dir": "/a/`id`"}, {"cache_dir": "/a/${USER}"},
    {"model_order": 3}, {"threads": 64, "cpus": 56}, {"families": ["nope"]}, {"families": ["freeform"]},
    {"action": "rm"}, {"cpus": "many"}, {"n_modes": 2.5},
])
def test_invalid_values_are_rejected(bad):
    with pytest.raises(TR.CommandError):
        TR.build_command(dict(FORM, **bad))


def test_api_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.service import api as A
    from src.service.model import ModelService
    c = TestClient(A.create_app(ModelService(None, "cpu"), datasets={}))
    cfg = c.get("/api/truba/config").json()
    assert cfg["form"]["labels"] == "field" and "elliptical" in cfg["families"]["train"]
    r = c.post("/api/truba/command", json=cfg["form"])
    assert r.status_code == 200 and "cluster/truba/run.sh all" in r.json()["command"]
    r = c.post("/api/truba/command", json=dict(cfg["form"], partition="x;y"))
    assert r.status_code == 422 and "partition" in r.json()["detail"]
