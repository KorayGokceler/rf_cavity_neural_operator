"""TRUBA one-line commands for the web UI's TRUBA page (cluster/truba/run.sh, docs/29 §5).

build_command(req) turns the page's form into ONE shell line to paste on a TRUBA login node:

    [ -d ~/rf_cavity_neural_operator ] || git clone <repo> ~/rf_cavity_neural_operator && \
    cd ~/rf_cavity_neural_operator && git fetch -q origin && git checkout -q <branch> && \
    git pull -q --ff-only origin <branch> && bash cluster/truba/run.sh <action> KEY=VALUE … [--dry-run]

Every value is validated (numbers by range, names by pattern) and shell-quoted, and only the
variables that differ from cluster/truba/config.sh's defaults are written, so the line stays short
and the cluster-side defaults stay the single source of truth.  Nothing here touches the cluster.
"""
import os
import re
import shlex
import subprocess

ACTIONS = ("all", "dataset", "train", "status", "setup")
TRAIN_FAMILIES = ("elliptical", "reentrant", "pillbox_pipes", "ridged_box", "composite", "freeform",
                  "hwr", "spoke", "dtl")
OOD_FAMILIES = ("box", "coax_qw", "pillbox_port", "elliptical_long", "junction")
MODELS = ("small", "base", "large", "xl")
DEFAULT_REPO = "https://github.com/KorayGokceler/rf_cavity_neural_operator.git"

# config.sh defaults (a value equal to its default is not written into the command)
DEFAULTS = {
    "LABELS": "n0", "PARTITION": "orfoz", "CPUS": 56, "TIME": "0-12:00:00", "ACCOUNT": "", "MAX_PARALLEL": 10,
    "MESH_SIZE": 0.10, "N_STORE": 10, "SAMPLING": "sobol", "DEFORM_PROB": 0.5, "DEFORM_MAX": 0.5, "SEED": 0,
    "LABEL_ORDER": 3, "MODEL_ORDER": 2, "MIN_FILLET": "", "THREADS": 8, "FIELD_MAX_ELEMENTS": 30000,
    "MAXH_FACTOR": 3.0, "ADAPT": 1, "TOL_F": 1e-6, "TOL_Q": 1e-3, "MODEL_MAX_ELEMENTS": 10000, "MAX_NDOF": 900000,
    "TAG": "", "TEST": 0,
    "GPU_PARTITION": "palamut-cuda", "GPUS": 1, "GPU_CPUS": 16, "GPU_TIME": "3-00:00:00",
    "EXP": "base", "MODEL": "base", "EPOCHS": 150, "LR": 3.0e-4, "CACHE_DIR": "", "TRAIN_EXTRA": "",
}
DATASET_KEYS = ("LABELS", "PARTITION", "CPUS", "TIME", "ACCOUNT", "MAX_PARALLEL", "MESH_SIZE", "N_STORE",
                "SAMPLING", "DEFORM_PROB", "DEFORM_MAX", "SEED", "TAG", "TEST")
FIELD_KEYS = ("LABEL_ORDER", "MODEL_ORDER", "MIN_FILLET", "THREADS", "FIELD_MAX_ELEMENTS", "MAXH_FACTOR", "ADAPT",
              "TOL_F", "TOL_Q", "MODEL_MAX_ELEMENTS", "MAX_NDOF")
TRAIN_KEYS = ("GPU_PARTITION", "GPUS", "GPU_CPUS", "GPU_TIME", "EXP", "MODEL", "EPOCHS", "BATCH", "LR",
              "NUM_WORKERS", "CACHE_DIR", "TRAIN_EXTRA", "TRAIN_FAMILIES")

_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$")
_SLURM_TIME = re.compile(r"^(\d{1,2}-)?\d{1,2}:\d{2}:\d{2}$")
_BRANCH = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._/\-]{0,127}$")
_REPO = re.compile(r"^(https://[A-Za-z0-9.\-]+(:\d+)?/[A-Za-z0-9._~/\-]+|git@[A-Za-z0-9.\-]+:[A-Za-z0-9._~/\-]+)$")
_DIR = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$")
_PATH = re.compile(r"^(/|\$HOME/|\$USER/)?([A-Za-z0-9_.\-]+|\$USER|\$HOME)(/([A-Za-z0-9_.\-]+|\$USER))*/?$")
_OVERRIDE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*=[A-Za-z0-9_.,:/\-\[\]\"']*$")

# rough cost model (docs/29 §4; cluster/truba/README.md for N0): core-seconds per geometry, MB on disk
FIELD_CORE_S = 560.0            # p3 label + p2 projection, ~18k curved tets (140 s on 4 threads)
SHARD_SIZE = 1024               # families.tsv shard_size (every family); TEST: one shard of TEST_N = 112
N0_CORE_S = {"elliptical": 2.2, "reentrant": 1.0, "pillbox_pipes": 1.2, "ridged_box": 1.0, "composite": 0.9,
             "freeform": 2.3, "hwr": 1.8, "spoke": 2.6, "dtl": 3.9}


def _hours(t):
    """SLURM [D-]HH:MM:SS → hours."""
    d, _, hms = t.rpartition("-")
    h, m, sec = (int(x) for x in hms.split(":"))
    return 24 * int(d or 0) + h + m / 60 + sec / 3600


class CommandError(ValueError):
    """An invalid form value (the API answers 422 with the message)."""


def repo_info(root):
    """(https clone URL, current branch) of the server's own checkout — the defaults of the page."""
    def git(*a):
        try:
            return subprocess.run(["git", "-C", root, *a], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    url = git("remote", "get-url", "origin")
    m = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(\.git)?$", url) or re.search(r"/git/([^/]+)/([^/]+?)(\.git)?$", url)
    repo = f"https://github.com/{m.group(1)}/{m.group(2)}.git" if m else DEFAULT_REPO
    return repo, git("rev-parse", "--abbrev-ref", "HEAD") or "main"


def _num(v, lo, hi, name, integer=False):
    try:
        x = int(v) if integer else float(v)
    except (TypeError, ValueError):
        raise CommandError(f"{name}: not a number ({v!r})") from None
    if integer and float(v) != x:
        raise CommandError(f"{name}: must be an integer")
    if not lo <= x <= hi:
        raise CommandError(f"{name}: {x} outside [{lo}, {hi}]")
    return x


def _pat(v, pat, name, allow_empty=True):
    v = "" if v is None else str(v).strip()
    if not v and allow_empty:
        return ""
    if not pat.match(v):
        raise CommandError(f"{name}: invalid value {v!r}")
    return v


def _families(sel, labels):
    sel = [str(f) for f in (sel or [])]
    bad = [f for f in sel if f not in TRAIN_FAMILIES + OOD_FAMILIES]
    if bad:
        raise CommandError(f"unknown families {bad}")
    if labels == "field":
        sel = [f for f in sel if f != "freeform"]          # field labels need a CAD solid
    if not sel:
        raise CommandError("choose at least one family")
    order = TRAIN_FAMILIES + OOD_FAMILIES
    sel = sorted(set(sel), key=order.index)
    if set(sel) == set(TRAIN_FAMILIES) - ({"freeform"} if labels == "field" else set()):
        return "train", sel
    if set(sel) == set(order) - ({"freeform"} if labels == "field" else set()):
        return "all", sel
    if set(sel) == set(OOD_FAMILIES):
        return "ood", sel
    return ",".join(sel), sel


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_command(req):
    """{'command', 'run_args', 'tag', 'families', 'estimate', 'notes'} for the page's form `req` (dict)."""
    r = dict(req or {})
    action = r.get("action", "all")
    if action not in ACTIONS:
        raise CommandError(f"action must be one of {ACTIONS}")
    labels = r.get("labels", "field")
    if labels not in ("field", "n0"):
        raise CommandError("labels: field or n0")
    v = {"LABELS": labels}
    v["PARTITION"] = _pat(r.get("partition", "orfoz"), _NAME, "partition", False)
    v["CPUS"] = _num(r.get("cpus", 56), 1, 1024, "cpus", True)
    v["TIME"] = _pat(r.get("time", "0-12:00:00"), _SLURM_TIME, "time", False)
    v["ACCOUNT"] = _pat(r.get("account", ""), _NAME, "account")
    v["MAX_PARALLEL"] = _num(r.get("max_parallel", 10), 1, 1000, "max_parallel", True)
    v["MESH_SIZE"] = _num(r.get("mesh_size", 0.10), 0.03, 0.5, "mesh_size")
    v["N_STORE"] = _num(r.get("n_modes", 10), 1, 40, "n_modes", True)
    v["SAMPLING"] = _pat(r.get("sampling", "sobol"), re.compile(r"^(sobol|random)$"), "sampling", False)
    v["DEFORM_PROB"] = _num(r.get("deform_prob", 0.5), 0.0, 1.0, "deform_prob")
    v["DEFORM_MAX"] = _num(r.get("deform_max", 0.5), 0.0, 0.9, "deform_max")
    v["SEED"] = _num(r.get("seed", 0), 0, 2**31 - 1, "seed", True)
    v["TAG"] = _pat(r.get("tag", ""), _NAME, "tag")
    v["TEST"] = 1 if r.get("test") else 0
    if labels == "field":
        v["LABEL_ORDER"] = _num(r.get("label_order", 3), 2, 4, "label_order", True)
        v["MODEL_ORDER"] = _num(r.get("model_order", 2), 1, 3, "model_order", True)
        if v["MODEL_ORDER"] >= v["LABEL_ORDER"]:
            raise CommandError("model_order must be below label_order (the model space is nested in the label space)")
        mf = r.get("min_fillet", 0.05)
        v["MIN_FILLET"] = "" if mf in (None, "") else _fmt(_num(mf, 0.0, 0.3, "min_fillet"))
        if v["MIN_FILLET"] == "0.05":
            v["MIN_FILLET"] = ""                            # the generator's field default
        v["THREADS"] = _num(r.get("threads", 8), 1, 128, "threads", True)
        if v["THREADS"] > v["CPUS"]:
            raise CommandError("threads per sample cannot exceed the cores per job")
        v["FIELD_MAX_ELEMENTS"] = _num(r.get("max_elements", 30000), 1000, 500000, "max_elements", True)
        v["MAXH_FACTOR"] = _num(r.get("maxh_factor", 3.0), 1.0, 10.0, "maxh_factor")
        v["ADAPT"] = 1 if r.get("adapt", True) else 0
        v["TOL_F"] = _num(r.get("tol_f", 1e-6), 1e-9, 1e-2, "tol_f")
        v["TOL_Q"] = _num(r.get("tol_q", 1e-3), 1e-6, 0.2, "tol_q")
        v["MODEL_MAX_ELEMENTS"] = _num(r.get("model_max_elements", 10000), 100, 500000, "model_max_elements", True)
        v["MAX_NDOF"] = _num(r.get("max_ndof", 900000), 10000, 20_000_000, "max_ndof", True)
    group, fams = _families(r.get("families") or list(TRAIN_FAMILIES), labels)
    n_per = r.get("n_per_family")
    n_per = None if n_per in (None, "") else _num(n_per, 1, 524288, "n_per_family", True)
    v["GPU_PARTITION"] = _pat(r.get("gpu_partition", "palamut-cuda"), _NAME, "gpu_partition", False)
    v["GPUS"] = _num(r.get("gpus", 1), 1, 16, "gpus", True)
    v["GPU_CPUS"] = _num(r.get("gpu_cpus", 16), 1, 256, "gpu_cpus", True)
    v["GPU_TIME"] = _pat(r.get("gpu_time", "3-00:00:00"), _SLURM_TIME, "gpu_time", False)
    v["EXP"] = _pat(r.get("exp", "field_base" if labels == "field" else "base"), _NAME, "exp", False)
    v["MODEL"] = r.get("model", "base")
    if v["MODEL"] not in MODELS:
        raise CommandError(f"model must be one of {MODELS}")
    v["EPOCHS"] = _num(r.get("epochs", 150), 1, 100000, "epochs", True)
    bt = r.get("batch")
    v["BATCH"] = "" if bt in (None, "") else _num(bt, 1, 1024, "batch", True)
    v["LR"] = _num(r.get("lr", 3e-4), 1e-7, 1.0, "lr")
    nw = r.get("num_workers")
    v["NUM_WORKERS"] = "" if nw in (None, "") else _num(nw, 0, 256, "num_workers", True)
    v["CACHE_DIR"] = _pat(r.get("cache_dir", ""), _PATH, "cache_dir") if labels == "field" else ""
    extra = [str(x).strip() for x in (r.get("train_extra") or []) if str(x).strip()]
    for x in extra:
        if not _OVERRIDE.match(x):
            raise CommandError(f"training override {x!r}: key=value with a plain value")
    v["TRAIN_EXTRA"] = " ".join(extra)
    tf = r.get("train_families")
    if labels == "field" and tf:
        _, tsel = _families(tf, labels)
        v["TRAIN_FAMILIES"] = " ".join(tsel)

    keys = {"setup": (), "status": ("LABELS", "MESH_SIZE", "N_STORE", "TAG", "TEST")
            + (FIELD_KEYS[:3] if labels == "field" else ()),
            "dataset": DATASET_KEYS + (FIELD_KEYS if labels == "field" else ()),
            "train": ("LABELS", "MESH_SIZE", "N_STORE", "TAG", "TEST") + (FIELD_KEYS[:3] if labels == "field" else ())
            + TRAIN_KEYS,
            "all": DATASET_KEYS + (FIELD_KEYS if labels == "field" else ()) + TRAIN_KEYS}[action]
    args = []
    if action in ("dataset", "train", "all") and group != "train":
        args.append(f"FAMILIES={group}")
    if action in ("dataset", "all") and n_per is not None:
        args.append(f"N_PER_FAMILY={n_per}")
    for k in keys:
        val = v.get(k, "")
        d = DEFAULTS.get(k, "")
        if val in ("", None) or (k != "LABELS" and _fmt(val) == _fmt(d)) or (k == "LABELS" and val == "n0"):
            continue
        sval = _fmt(val)
        # paths may name $USER / $HOME (validated by _PATH): double quotes let the login shell expand them
        args.append(f'{k}="{sval}"' if "$" in sval else f"{k}={shlex.quote(sval)}")
    if r.get("dry_run"):
        args.append("--dry-run")

    repo = _pat(r.get("repo") or DEFAULT_REPO, _REPO, "repo", False)
    branch = _pat(r.get("branch") or "main", _BRANCH, "branch", False)
    rdir = _pat(r.get("repo_dir") or "rf_cavity_neural_operator", _DIR, "repo_dir", False)
    home = f"~/{rdir}"
    run = " ".join(["bash cluster/truba/run.sh", action] + args)
    command = (f"[ -d {home} ] || git clone {shlex.quote(repo)} {home}; cd {home} && git fetch -q origin && "
               f"git checkout -q {shlex.quote(branch)} && git pull -q --ff-only origin {shlex.quote(branch)} && {run}")

    # TAG as config.sh will derive it
    if v["TAG"]:
        tag = v["TAG"]
    elif labels == "field":
        tag = (f"F_p{v['LABEL_ORDER']}p{v['MODEL_ORDER']}_ms{_fmt(v['MESH_SIZE'])}_k{v['N_STORE']}"
               f"_mf{v['MIN_FILLET'] or '0.05'}_v1")
    else:
        tag = f"E_ms{_fmt(v['MESH_SIZE'])}_k{v['N_STORE']}_v1"
    tag = tag.replace("_ms0.1_", "_ms0.10_")            # MESH_SIZE left at config.sh's literal 0.10
    if v["TEST"]:
        tag += "_test"

    n_geo = (112 if v["TEST"] else (n_per or 10240)) * len(fams)
    if labels == "field":
        core_h = n_geo * FIELD_CORE_S * (v["N_STORE"] / 10) ** 0.3 * (v["LABEL_ORDER"] / 3) ** 3 / 3600
        if v["ADAPT"]:
            core_h *= 3.0                    # ~3 refinement levels of a growing mesh (docs/30 §6)
        size = min(v["MODEL_MAX_ELEMENTS"], 20000) / 20000 if v["ADAPT"] else 1.0
        mb = n_geo * (1.0 + 0.8 * v["N_STORE"] * size
                      * (1 if v["MODEL_ORDER"] == 2 else 0.27 if v["MODEL_ORDER"] == 1 else 2.2))
    else:
        core_h = sum((112 if v["TEST"] else (n_per or 10240)) * N0_CORE_S.get(f, 1.5) for f in fams) \
            * (0.1 / v["MESH_SIZE"]) ** 2.5 / 3600
        mb = n_geo * 1.8
    workers = max(1, v["CPUS"] // v["THREADS"]) if labels == "field" else v["CPUS"]
    notes = []
    if labels == "field" and v["MODEL_ORDER"] == 1:
        notes.append("model_order1")
    if labels == "field" and "ridged_box" in fams:
        notes.append("ridged_box_cost")
    if action in ("all", "train") and labels == "field" and not v["CACHE_DIR"]:
        notes.append("no_cache")
    if action in ("dataset", "all") and labels == "field":
        shard_h = core_h / n_geo * (112 if v["TEST"] else SHARD_SIZE) / (workers * v["THREADS"])
        if shard_h > 0.8 * _hours(v["TIME"]):
            notes.append("shard_time")     # a killed shard resumes on the next run, but training may start first
    return {"command": command, "run_args": run, "tag": tag, "families": fams, "group": group,
            "estimate": {"geometries": n_geo, "core_hours": round(core_h, 1), "disk_gb": round(mb / 1024, 1),
                         "workers_per_job": workers},
            "notes": notes}


def defaults(root):
    """The page's initial form + choices."""
    repo, branch = repo_info(root)
    return {"repo": repo, "branch": branch, "repo_dir": "rf_cavity_neural_operator", "actions": list(ACTIONS),
            "families": {"train": list(TRAIN_FAMILIES), "ood": list(OOD_FAMILIES)}, "models": list(MODELS),
            "form": {"action": "all", "labels": "field", "families": [f for f in TRAIN_FAMILIES if f != "freeform"],
                     "n_per_family": 1024, "mesh_size": 0.10, "n_modes": 10, "label_order": 3, "model_order": 2,
                     "min_fillet": 0.05, "threads": 8, "max_elements": 30000, "maxh_factor": 3.0,
                     "adapt": True, "tol_f": 1e-6, "tol_q": 1e-3, "model_max_elements": 10000, "max_ndof": 900000,
                     "deform_prob": 0.5, "deform_max": 0.5, "sampling": "sobol", "seed": 0, "tag": "",
                     "partition": "orfoz", "cpus": 56, "time": "0-12:00:00", "account": "", "max_parallel": 10,
                     "test": False, "gpu_partition": "palamut-cuda", "gpus": 1, "gpu_cpus": 16,
                     "gpu_time": "3-00:00:00", "exp": "field_base", "model": "base", "epochs": 150, "batch": "",
                     "lr": 3e-4, "num_workers": "", "cache_dir": "", "train_extra": [], "dry_run": False,
                     "repo": repo, "branch": branch, "repo_dir": "rf_cavity_neural_operator"}}
