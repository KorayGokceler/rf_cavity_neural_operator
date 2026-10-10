"""Web API of the cavity eigenmode predictor (FastAPI). docs/26_WEB_UI.md.

    RFCAV_CHECKPOINT=runs/large uvicorn src.service.api:app --port 8000
    (no RFCAV_CHECKPOINT: an UNTRAINED demo model — for UI development only)

Endpoints (JSON; large arrays as base64 little-endian float32 / uint32, key suffix _b64):
    GET  /api/info                               model, families, limits
    POST /api/geometries            (multipart)  upload CAD / mesh → checks + display surface
    POST /api/geometries/sample     {family,id}  a generator geometry → checks + display surface
    POST /api/predictions           {geometry_id}           mode table + figures of merit
    GET  /api/predictions/{id}/modes/{k}          surface E/H + beam-axis E_z of mode k
    GET  /api/predictions/{id}/plane?mode&axis&pos&res      cut plane E/H of mode k
    GET  /api/predictions/{id}/export             everything as one JSON file
    GET  /api/geometries/{id}/features            model input features (names, ranges)
    GET  /api/geometries/{id}/feature_plane|feature_surface?name&…   one feature on a cut / the wall
    GET  /api/datasets                            registered datasets (RFCAV_DATA, src/service/datasets.py)
    GET  /api/datasets/{d}/items | /stats         geometry table (filter, sort, page) / overview
    POST /api/datasets/{d}/items/{id}/open        geometry + FE solution (+ labels, parameters)
    POST /api/predictions {geometry_id, compare_to}  + model-vs-FE rows and a difference solution
    GET  /api/jobs/config · POST /api/jobs/generate · GET /api/jobs[/{id}] · POST /api/jobs/{id}/cancel
    GET  /api/train/config · GET /api/train/runs[/{name}] · POST /api/train/start · /runs/{name}/resume
    POST /api/train/stop · POST /api/model/load {run}   training from the UI (RFCAV_RUNS_ROOT)
    GET  /api/truba/config · POST /api/truba/command {form}   one-line TRUBA command (src/service/truba.py)
The built frontend (web/dist) is served at / when present.
"""
import base64
import collections
import glob
import hashlib
import hmac
import os
import random
import re
import tempfile
import threading
import time
import uuid

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data.dataset_converter_3d import FEATURE_NAMES_3D
from src.service import geometry as G
from src.service import training as T
from src.service import truba as TR

VERSION = '0.1.0'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_FAMILIES = ('elliptical', 'reentrant', 'pillbox_pipes', 'ridged_box', 'composite', 'freeform',
                  'hwr', 'spoke', 'dtl')
OOD_FAMILIES = ('box', 'coax_qw', 'pillbox_port', 'elliptical_long', 'junction')


def _env(name, default, cast=float):
    v = os.environ.get(name)
    return cast(v) if v not in (None, '') else default


class Limits:
    max_upload_mb = _env('RFCAV_MAX_UPLOAD_MB', 25.0)
    mesh_timeout_s = _env('RFCAV_MESH_TIMEOUT', 120.0)
    mesh_mem_mb = _env('RFCAV_MESH_MEM_MB', 4096, int)
    max_tets = _env('RFCAV_MAX_TETS', 400_000, int)
    min_mesh_size = 0.05
    max_items = _env('RFCAV_MAX_ITEMS', 64, int)
    ttl_s = _env('RFCAV_TTL_S', 3600.0)


class Store:
    """In-memory LRU with expiry (v0.1: single process; nothing is written to disk)."""

    def __init__(self, max_items, ttl_s):
        self.max_items, self.ttl_s = max_items, ttl_s
        self._d = collections.OrderedDict()
        self._lock = threading.Lock()

    def put(self, value):
        key = uuid.uuid4().hex
        with self._lock:
            self._d[key] = (time.time(), value)
            self._evict()
        return key

    def get(self, key):
        with self._lock:
            self._evict()
            if key not in self._d:
                raise HTTPException(404, 'not found or expired')
            t, v = self._d.pop(key)
            self._d[key] = (t, v)
            return v

    def _evict(self):
        now = time.time()
        for k in [k for k, (t, _) in self._d.items() if now - t > self.ttl_s]:
            del self._d[k]
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)


def b64(a, dtype=np.float32):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode('ascii')


def _clean(v):
    return None if v is None or not np.isfinite(v) else float(v)


class SampleReq(BaseModel):
    family: str
    id: int | None = Field(default=None, ge=0, lt=2 ** 24)
    mesh_size: float = Field(default=0.10, ge=Limits.min_mesh_size, le=0.5)


class PredictReq(BaseModel):
    geometry_id: str
    compare_to: str | None = None          # FE solution id of the same geometry (dataset item)


TOKEN_COOKIE = 'rfcav_token'


def _access_token(app, token):
    """Optional shared-secret access (RFCAV_TOKEN) for a server reachable from outside (tunnel, LAN):
    open the page once as /?token=<token> — the token is then kept in an HttpOnly cookie; API calls
    without it get 401. No token configured → open access (local development)."""
    if not token:
        return

    def ok(v):
        return bool(v) and hmac.compare_digest(str(v), token)

    @app.middleware('http')
    async def guard(request, call_next):
        q = request.query_params.get('token')
        if ok(q) or ok(request.cookies.get(TOKEN_COOKIE)) or ok(request.headers.get('x-rfcav-token')):
            resp = await call_next(request)
            if ok(q):
                resp.set_cookie(TOKEN_COOKIE, token, httponly=True, samesite='lax',
                                secure=request.url.scheme == 'https', max_age=7 * 24 * 3600)
            return resp
        if request.url.path.startswith('/api/'):
            return JSONResponse({'detail': 'access token required'}, status_code=401)
        return HTMLResponse('<!doctype html><meta name="viewport" content="width=device-width">'
                            '<body style="font-family:system-ui;padding:24px">'
                            '<h3>RF Cavity Neural Solver</h3><p>Erişim anahtarı gerekli: sunucunun verdiği '
                            'bağlantıyı (…/?token=…) açın. · Access token required: open the link with '
                            '?token=… printed by the server.</p></body>', status_code=401)


def create_app(service=None, datasets=None):
    """FastAPI app around a ModelService (built from RFCAV_CHECKPOINT / RFCAV_DEVICE when None) and
    a dataset registry ({name: dataset}; RFCAV_DATA when None)."""
    from src.service.datasets import discover
    fixed = datasets                                        # tests pass a registry; else RFCAV_DATA
    gen_root = os.environ.get('RFCAV_GEN_ROOT') or None
    ds_ids = {}

    def _same(a, b):
        if type(a) is not type(b):
            return False
        if hasattr(a, 'files'):
            return a.files == b.files
        return a.path == b.path and os.path.getmtime(a.path) == getattr(a, '_mtime', None)

    def rescan():
        """(Re)build the registry: RFCAV_DATA (or the fixed one) + generated datasets under RFCAV_GEN_ROOT.
        Ids are stable per name; unchanged datasets keep their loaded state."""
        found = dict(fixed) if fixed is not None else discover()
        if gen_root:                                        # a data root may contain gen_root: list it once
            under = os.path.abspath(gen_root) + os.sep
            found = {k: d for k, d in found.items()
                     if not any(os.path.abspath(f).startswith(under) for f in getattr(d, 'files', [getattr(d, 'path', '')]))}
        if gen_root and os.path.isdir(gen_root):
            found.update({f"generated/{k}": d for k, d in discover(gen_root).items()})
        old = {d.name: d for d in ds_ids.values()}
        new = {}
        for name, d in found.items():
            d.name = name
            if hasattr(d, 'path'):
                d._mtime = os.path.getmtime(d.path)
            keep = old.get(name)
            new['d' + hashlib.sha1(name.encode()).hexdigest()[:8]] = keep if keep is not None and _same(keep, d) else d
        ds_ids.clear()
        ds_ids.update(new)

    rescan()
    if service is None:
        from src.service.model import ModelService
        service = ModelService(os.environ.get('RFCAV_CHECKPOINT') or None, os.environ.get('RFCAV_DEVICE') or None,
                               optional=os.environ.get('RFCAV_CHECKPOINT_OPTIONAL') == '1')
    lim = Limits()
    geoms, preds = Store(lim.max_items, lim.ttl_s), Store(lim.max_items, lim.ttl_s)
    app = FastAPI(title='RF Cavity Neural Solver', version=VERSION)
    app.state.service = service
    _access_token(app, os.environ.get('RFCAV_TOKEN') or None)

    def _geometry_response(nodes, tets, source, extra=None, prepared=None):
        if len(tets) > lim.max_tets:
            raise HTTPException(413, f"mesh has {len(tets)} tets (limit {lim.max_tets}); use a coarser mesh size")
        chk = G.check(nodes, tets)
        vid, tri = G.surface(nodes, tets)
        rec = {'nodes': nodes, 'tets': tets, 'vid': vid, 'tri': tri, 'source': source,
               'family': (extra or {}).get('family', ''), 'check': chk}
        if prepared:
            rec.update(prepared)
        gid = geoms.put(rec)
        pts = np.asarray(nodes)[vid] * 1e3
        return {'id': gid, 'source': source, 'check': chk, **(extra or {}),
                'surface': {'n_points': int(len(vid)), 'n_triangles': int(len(tri)),
                            'points_b64': b64(pts), 'triangles_b64': b64(tri, np.uint32)}}

    @app.get('/api/info')
    def info():
        return {'version': VERSION, 'model': service.info(),
                'families': {'train': list(TRAIN_FAMILIES), 'ood': list(OOD_FAMILIES)},
                'limits': {'max_upload_mb': lim.max_upload_mb, 'max_tets': lim.max_tets,
                           'cad': list(G.CAD_EXT), 'mesh': list(G.MESH_EXT)},
                'conventions': {'U_J': 1.0, 'wall': 'copper, sigma = 5.8e7 S/m', 'beta': 1.0,
                                'R_over_Q': 'linac: V^2/(omega U)', 'beam_axis': 'x = y = 0 along z (hwr: x)'}}

    @app.post('/api/geometries')
    def upload(file: UploadFile = File(...), unit: str = Form('mm'), mesh_size: float = Form(0.10),
               n_cells: float = Form(1.0)):
        if unit not in G.UNITS:
            raise HTTPException(422, f"unit must be one of {sorted(G.UNITS)}")
        if not (lim.min_mesh_size <= mesh_size <= 0.5) or not (1 <= n_cells <= 30):
            raise HTTPException(422, 'mesh_size must be in [0.05, 0.5], n_cells in [1, 30]')
        ext = os.path.splitext(file.filename or '')[1].lower()
        if ext not in G.CAD_EXT + G.MESH_EXT:
            raise HTTPException(415, f"unsupported file type {ext!r}")
        data = file.file.read(int(lim.max_upload_mb * 2 ** 20) + 1)
        if len(data) > lim.max_upload_mb * 2 ** 20:
            raise HTTPException(413, f"file larger than {lim.max_upload_mb:g} MB")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'upload' + ext)                 # never the client's file name
            with open(path, 'wb') as fo:
                fo.write(data)
            try:
                nodes, tets = G.run_isolated(G.mesh_upload, path, unit, mesh_size, n_cells, None,
                                             timeout=lim.mesh_timeout_s, mem_mb=lim.mesh_mem_mb)
            except (RuntimeError, TimeoutError) as e:
                raise HTTPException(422, f"could not mesh the file: {e}") from None
        return _geometry_response(nodes, tets, {'kind': 'upload', 'name': os.path.basename(file.filename or ''),
                                                'unit': unit, 'mesh_size': mesh_size, 'n_cells': n_cells})

    @app.post('/api/geometries/sample')
    def sample(req: SampleReq):
        if req.family not in TRAIN_FAMILIES + OOD_FAMILIES:
            raise HTTPException(422, f"unknown family {req.family!r}")
        s_id = req.id if req.id is not None else random.randrange(2 ** 20, 2 ** 21)   # beyond the training ids
        try:
            nodes, tets, params = G.run_isolated(G.mesh_family, req.family, s_id, req.mesh_size,
                                                 timeout=lim.mesh_timeout_s, mem_mb=lim.mesh_mem_mb)
        except (RuntimeError, TimeoutError) as e:
            raise HTTPException(422, f"could not build the geometry: {e}") from None
        return _geometry_response(nodes, tets, {'kind': 'sample', 'family': req.family, 'id': s_id,
                                                'mesh_size': req.mesh_size, 'ood': req.family in OOD_FAMILIES},
                                  {'family': req.family, 'params': params})

    def _prepared(g):
        """Converter geometry (features + operators) of a stored geometry, computed once."""
        if '_geom' not in g:
            g['_geom'], g['_M'] = service.geometry(g['nodes'], g['tets'], g['family'])
        return g['_geom'], g['_M']

    def _summary(sid, sol, extra=None):
        return {'id': sid, 'source': sol['source'], 'n_edges': sol['n_edges'],
                'modes': sol['modes'], 'axis_length_mm': sol['axis_length_mm'], **(extra or {})}

    @app.post('/api/predictions')
    def predict(req: PredictReq):
        g = geoms.get(req.geometry_id)
        if not g['check']['ok']:
            raise HTTPException(422, 'geometry failed the checks: ' + '; '.join(g['check']['problems']))
        t0 = time.perf_counter()
        geom, M = _prepared(g)
        p = service.predict(g['nodes'], g['tets'], g['family'], geom=geom, M=M)
        p['time_s']['total'] = time.perf_counter() - t0
        p['_geometry_id'] = req.geometry_id
        pid = preds.put(p)
        warnings = []
        if service.untrained:
            warnings.append('untrained demo model: the numbers are meaningless (UI development mode)'
                            + (f" — the checkpoint did not load ({service.load_error})" if getattr(service, 'load_error', None) else ''))
        if p['axis_length_mm'] <= 0:
            warnings.append('the beam axis (x = y = 0 along z) does not cross the cavity: R/Q, R_sh, T, '
                            'Epk/Eacc and Bpk/Eacc are undefined')
        if g['source'].get('ood'):
            warnings.append('out-of-distribution family: verify with a full-wave solver')
        out = _summary(pid, p, {'geometry_id': req.geometry_id, 'time_s': p['time_s'], 'warnings': warnings,
                                'model': service.info()})
        if req.compare_to:
            from src.service.model import compare
            truth = preds.get(req.compare_to)
            if truth.get('_geometry_id') != req.geometry_id:
                raise HTTPException(422, 'compare_to must be the FE solution of the same geometry')
            rows, diff = compare(p, truth)
            diff['_geometry_id'] = req.geometry_id
            out['comparison'] = {'rows': rows, 'diff_id': preds.put(diff), 'truth_id': req.compare_to}
        return out

    @app.get('/api/predictions/{pid}/modes/{k}')
    def mode(pid: str, k: int):
        p = preds.get(pid)
        if not 0 <= k < len(p['_f']):
            raise HTTPException(404, 'no such mode')
        g = geoms.get(p['_geometry_id'])
        E, H = service.surface_fields(p, g['vid'], k)
        z, ez = service.axis_profile(p, k)
        return {'mode': k, 'f_GHz': float(p['_f'][k]),
                'surface': {'E_b64': b64(E), 'H_b64': b64(H),
                            'E_max': float(np.linalg.norm(p['_E'][:, :, k], axis=1).max()),
                            'H_max': float(np.linalg.norm(p['_H'][:, :, k], axis=1).max())},
                'axis': {'z_mm': [float(v) for v in z], 'Ez': [_clean(v) for v in ez]}}

    @app.get('/api/predictions/{pid}/plane')
    def plane(pid: str, mode: int = 0, axis: str = 'y', pos: float | None = None, res: int = 121):
        p = preds.get(pid)
        if axis not in ('x', 'y', 'z') or not 16 <= res <= 241 or not 0 <= mode < len(p['_f']):
            raise HTTPException(422, 'axis in x|y|z, res in [16, 241], valid mode')
        pts, tri, E, H = service.plane(p, mode, axis, pos, res)
        return {'mode': mode, 'axis': axis, 'n_points': int(len(pts)), 'n_triangles': int(len(tri)),
                'points_b64': b64(pts), 'triangles_b64': b64(tri, np.uint32), 'E_b64': b64(E), 'H_b64': b64(H)}

    @app.get('/api/predictions/{pid}/export')
    def export(pid: str):
        p = preds.get(pid)
        g = geoms.get(p['_geometry_id'])
        body = {'version': VERSION, 'model': service.info(), 'geometry': {'source': g['source'], 'check': g['check']},
                'source': p['source'], 'modes': p['modes'], 'time_s': p.get('time_s'),
                'conventions': info()['conventions']}
        return JSONResponse(body, headers={'Content-Disposition': f'attachment; filename="prediction_{pid[:8]}.json"'})

    # ── model input features ─────────────────────────────────────────
    def _feature(g, name):
        geom, _ = _prepared(g)
        names = list(g.get('feature_names') or FEATURE_NAMES_3D)
        if name not in names:
            raise HTTPException(404, f"unknown feature {name!r}")
        return geom, np.asarray(geom['Input_funcs'], float)[:, names.index(name)]

    @app.get('/api/geometries/{gid}/features')
    def features(gid: str):
        g = geoms.get(gid)
        geom, _ = _prepared(g)
        names = list(g.get('feature_names') or FEATURE_NAMES_3D)
        F = np.asarray(geom['Input_funcs'], float)
        return {'names': names, 'min': F.min(0).tolist(), 'max': F.max(0).tolist(),
                'n_vertices': int(len(F)), 'n_edges': int(len(geom['edges'])),
                'scale_mm': float(geom['scale'] * 1e3), 'torsion_max': float(geom.get('torsion_max', 0.0))}

    @app.get('/api/geometries/{gid}/feature_plane')
    def feature_plane(gid: str, name: str, axis: str = 'y', pos: float | None = None, res: int = 121):
        if axis not in ('x', 'y', 'z') or not 16 <= res <= 241:
            raise HTTPException(422, 'axis in x|y|z, res in [16, 241]')
        geom, v = _feature(geoms.get(gid), name)
        pts, tri, (vals,) = service.plane_values(geom, [v], axis, pos, res)
        return {'name': name, 'n_points': int(len(pts)), 'n_triangles': int(len(tri)), 'points_b64': b64(pts),
                'triangles_b64': b64(tri, np.uint32), 'values_b64': b64(vals)}

    @app.get('/api/geometries/{gid}/feature_surface')
    def feature_surface(gid: str, name: str):
        g = geoms.get(gid)
        _, v = _feature(g, name)
        return {'name': name, 'values_b64': b64(v[g['vid']])}

    @app.get('/api/geometries/{gid}/feature_view')
    def feature_view_ep(gid: str, kind: str, channel: str | None = None, axis: str = 'y',
                        pos: float | None = None, res: int = 121):
        from src.service.features import KINDS, feature_view
        if kind not in KINDS or axis not in ('x', 'y', 'z') or not 16 <= res <= 241:
            raise HTTPException(422, f"kind in {KINDS}, axis in x|y|z, res in [16, 241]")
        g = geoms.get(gid)
        geom, _ = _prepared(g)
        try:
            v = feature_view(geom, kind, axis, pos, res, g.get('feature_names'), channel)
        except KeyError:
            raise HTTPException(404, f"unknown feature channel {channel!r}") from None

        def mesh(m):
            return None if m is None else {'n_points': int(len(m['points'])), 'points_b64': b64(m['points']),
                                           'triangles_b64': b64(m['triangles'], np.uint32),
                                           'scalars_b64': b64(np.nan_to_num(m['scalars']))}
        return {'kind': kind, 'plane': mesh(v['plane']), 'cells': mesh(v['cells']),
                'segments': [{'role': sg['role'], 'n': int(len(sg['points'])), 'points_b64': b64(sg['points'])}
                             for sg in v['segments']],
                'range': v['range'], 'signed': v['signed'], 'neutral': bool(v.get('neutral', False)),
                'legend': v['legend']}

    # ── datasets ─────────────────────────────────────────────────────
    def _ds(did):
        if did not in ds_ids:
            raise HTTPException(404, 'unknown dataset')
        return ds_ids[did]

    def _rows(d):
        try:
            return d.rows()
        except MemoryError as e:
            raise HTTPException(413, str(e)) from None

    @app.get('/api/datasets')
    def datasets_list():
        return [{'id': i, 'name': d.name, 'kind': d.kind} for i, d in ds_ids.items()]

    @app.get('/api/datasets/{did}/items')
    def dataset_items(did: str, family: str | None = None, offset: int = 0, limit: int = 50,
                      sort: str = 'id', desc: bool = False):
        rows = _rows(_ds(did))
        if family:
            rows = [r for r in rows if r['family'] == family]
        keyf = {'id': lambda r: r['id'], 'family': lambda r: r['family'], 'n_edges': lambda r: r['n_edges'],
                'n_nodes': lambda r: r['n_nodes'], 'betti1': lambda r: r['betti1'],
                'f0': lambda r: r['f_GHz'][0] if r['f_GHz'] else np.inf}.get(sort)
        if keyf is None:
            raise HTTPException(422, 'sort by id | family | n_edges | n_nodes | betti1 | f0')
        rows = sorted(rows, key=keyf, reverse=desc)
        limit = max(1, min(int(limit), 500))
        return {'total': len(rows), 'offset': offset, 'items': rows[offset:offset + limit]}

    @app.get('/api/datasets/{did}/stats')
    def dataset_stats(did: str):
        rows = _rows(_ds(did))
        fam = collections.defaultdict(lambda: {'count': 0, 'f0': [], 'n_edges': []})
        for r in rows:
            e = fam[r['family']]
            e['count'] += 1
            e['n_edges'].append(r['n_edges'])
            if r['f_GHz']:
                e['f0'].append(r['f_GHz'][0])
        return {'total': len(rows), 'families': dict(sorted(fam.items())),
                'n_modes': max((len(r['f_GHz']) for r in rows), default=0)}

    @app.post('/api/datasets/{did}/items/{sid}/open')
    def dataset_open(did: str, sid: int):
        d = _ds(did)
        try:
            r = d.open(sid)
        except KeyError:
            raise HTTPException(404, 'no such geometry') from None
        except MemoryError as e:
            raise HTTPException(413, str(e)) from None
        from src.data.dataset_3d import to_csr
        ne = len(r['geom']['edges'])
        M = to_csr(r['M'], (ne, ne))
        resp = _geometry_response(r['nodes'], r['tets'],
                                  {'kind': 'dataset', 'dataset': d.name, 'family': r['family'], 'id': r['id'],
                                   'file': r['source_file']},
                                  {'family': r['family'], 'params': r['params']},
                                  prepared={'_geom': r['geom'], '_M': M, 'feature_names': r['features']})
        truth = None
        if r['U'].shape[1]:
            from src.service.model import build_solution
            sol = build_solution(r['geom'], M, r['U'], r['f_GHz'], 'fe')
            sol['_geometry_id'] = resp['id']
            truth = _summary(preds.put(sol), sol, {'f_next': r['f_next'], 'labels': r['qoi_labels']})
        resp.update(truth=truth)
        return resp

    @app.post('/api/datasets/rescan')
    def datasets_rescan():
        rescan()
        return datasets_list()

    # ── dataset generation (background jobs; RFCAV_GEN_ROOT, e.g. a Google Drive folder) ──
    from src.service import jobs as J
    blocks = J.family_blocks()
    jm = J.JobManager(gen_root, on_done=lambda job: rescan()) if gen_root else None
    max_gen = _env('RFCAV_GEN_MAX', 20000, int)

    class GenReq(BaseModel):
        families: list[str]
        n_total: int = Field(default=100, ge=1)
        mesh_size: float = Field(default=0.10, ge=Limits.min_mesh_size, le=0.5)
        n_modes: int = Field(default=10, ge=1, le=30)
        sampling: str = Field(default='sobol', pattern='^(sobol|random)$')
        deform_prob: float = Field(default=0.5, ge=0.0, le=1.0)
        deform_max: float = Field(default=0.5, ge=0.0, le=0.9)
        seed: int = Field(default=0, ge=0)
        shard_size: int = Field(default=256, ge=16, le=4096)
        workers: int | None = Field(default=None, ge=1, le=256)
        convert: bool = True
        lean: bool = True
        tag: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$')

    def _jm():
        if jm is None:
            raise HTTPException(503, 'dataset generation is off: start the server with RFCAV_GEN_ROOT=<folder> '
                                     '(e.g. a Google Drive folder)')
        return jm

    @app.get('/api/jobs/config')
    def jobs_config():
        return {'enabled': jm is not None, 'gen_root': gen_root,
                'cpu_count': os.cpu_count() or 1, 'max_total': max_gen,
                'families': {'train': [f for f in TRAIN_FAMILIES], 'ood': [f for f in OOD_FAMILIES]},
                'blocks': {f: blocks.get(f) for f in TRAIN_FAMILIES + OOD_FAMILIES},
                'tag_example': J.tag_of(0.10, 10)}

    @app.post('/api/jobs/generate')
    def jobs_generate(req: GenReq):
        m = _jm()
        if not req.families:
            raise HTTPException(422, 'choose at least one family')
        bad = [f for f in req.families if f not in TRAIN_FAMILIES + OOD_FAMILIES or blocks.get(f) is None]
        if bad:
            raise HTTPException(422, f"unknown family / no id block: {bad}")
        if req.n_total > max_gen:
            raise HTTPException(422, f"n_total ≤ {max_gen} per family (RFCAV_GEN_MAX)")
        tag = req.tag or J.tag_of(req.mesh_size, req.n_modes)
        out = []
        for f in req.families:
            p = req.model_dump(exclude={'families', 'tag', 'workers'})
            p.update(family=f, block=blocks[f], tag=tag,
                     workers=req.workers or os.cpu_count() or 1,
                     group='ood' if f in OOD_FAMILIES else 'train')
            out.append(m.submit(p).public())
        return out

    @app.get('/api/jobs')
    def jobs_list():
        return [j.public() for j in reversed(_jm().jobs.values())] if jm else []

    @app.get('/api/jobs/{jid}')
    def jobs_get(jid: str):
        m = _jm()
        if jid not in m.jobs:
            raise HTTPException(404, 'no such job')
        return m.jobs[jid].public()

    @app.post('/api/jobs/{jid}/cancel')
    def jobs_cancel(jid: str):
        m = _jm()
        if jid not in m.jobs:
            raise HTTPException(404, 'no such job')
        return m.cancel(jid).public()

    # ── training (src/service/training.py) ────────────────────────────
    runs_root = os.environ.get('RFCAV_RUNS_ROOT') or None
    tm = T.TrainManager(runs_root) if runs_root else None

    class TrainReq(BaseModel):
        name: str = Field(pattern=T.NAME_RE)
        datasets: list[str] = Field(default_factory=list)
        preset: str = Field(default='small', pattern='^(' + '|'.join(T.PRESETS) + ')$')
        epochs: int = Field(default=100, ge=1, le=10000)
        batch_size: int = Field(default=4, ge=1, le=256)
        lr: float = Field(default=2e-4, gt=0, le=1.0)
        n_modes: int = Field(default=6, ge=1, le=30)
        patience: int = Field(default=50, ge=1, le=10000)
        qoi_weight: float = Field(default=0.0, ge=0.0, le=100.0)
        cache_operators: bool | None = None

    class ResumeReq(BaseModel):
        epochs: int | None = Field(default=None, ge=1, le=10000)

    class LoadReq(BaseModel):
        run: str = Field(pattern=T.NAME_RE)

    def _tm():
        if tm is None:
            raise HTTPException(503, 'training is off: start the server with RFCAV_RUNS_ROOT=<folder> '
                                     '(e.g. <Drive>/training_logs)')
        return tm

    def _h5_of(d):
        """H5 shards of a registered dataset: its own files, or for a generated / TRUBA family PKL
        (<TAG>/pkl/<fam>.pkl) the shards in <TAG>/h5/<fam>/."""
        if getattr(d, 'files', None):
            return list(d.files)
        p = getattr(d, 'path', '')
        base, fam = os.path.dirname(os.path.dirname(p)), os.path.splitext(os.path.basename(p))[0]
        if os.path.basename(os.path.dirname(p)) == 'pkl':
            return sorted(glob.glob(os.path.join(base, 'h5', fam, '*.h5')))
        return []

    def _train_data(ids):
        if not ids:
            raise HTTPException(422, 'choose at least one dataset')
        ds = [_ds(i) for i in ids]
        if len(ds) == 1 and getattr(ds[0], 'path', None):
            return {'pkl': ds[0].path}, [ds[0].name]
        files, missing = set(), []
        for d in ds:
            h = _h5_of(d)
            (files.update(h) if h else missing.append(d.name))
        if missing:
            raise HTTPException(422, f"no H5 shards for {missing}: several datasets are merged from their H5 "
                                     "shards — choose a PKL alone, or its H5 folder")
        return {'h5': sorted(files)}, [d.name for d in ds]

    def _gpu():
        try:
            import torch
            return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:                                    # noqa: BLE001
            return None

    # ── TRUBA one-line commands (src/service/truba.py; nothing is sent to the cluster) ──
    @app.get('/api/truba/config')
    def truba_config():
        return TR.defaults(ROOT)

    @app.post('/api/truba/command')
    def truba_command(form: dict = Body(...)):
        try:
            return TR.build_command(form)
        except TR.CommandError as e:
            raise HTTPException(422, str(e)) from None

    @app.get('/api/train/config')
    def train_config():
        return {'enabled': tm is not None, 'runs_root': runs_root, 'gpu': _gpu(), 'cpu_count': os.cpu_count() or 1,
                'presets': {k: {'embed_dim': v[0], 'n_heads': v[1], 'n_layers': v[2], 'n_basis': v[3],
                                'params_m': T.PRESET_PARAMS_M[k]} for k, v in T.PRESETS.items()},
                'busy': bool(tm and tm.busy()), 'active': tm.active.name if tm and tm.active else None,
                'model': service.info()}

    @app.get('/api/train/runs')
    def train_runs():
        return _tm().runs() if tm else []

    @app.get('/api/train/runs/{name}')
    def train_run(name: str):
        try:
            return _tm().detail(name)
        except KeyError:
            raise HTTPException(404, 'no such run') from None

    def _start(name, params, data, resume):
        try:
            _tm().start(name, params, data, resume=resume)
        except (RuntimeError, FileExistsError) as e:
            raise HTTPException(409, str(e)) from None
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from None
        return tm.detail(name)

    @app.post('/api/train/start')
    def train_start(req: TrainReq):
        _tm()
        data, names = _train_data(req.datasets)
        params = req.model_dump(exclude={'name', 'datasets'})
        params['sources'] = names
        return _start(req.name, params, data, False)

    @app.post('/api/train/runs/{name}/resume')
    def train_resume(name: str, req: ResumeReq):
        if not re.match(T.NAME_RE, name):
            raise HTTPException(422, 'bad run name')
        try:
            p = _tm().resume_params(name)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from None
        data = p.pop('data')
        if req.epochs:
            p['epochs'] = req.epochs
        return _start(name, p, data, True)

    @app.post('/api/train/stop')
    def train_stop():
        r = _tm().stop()
        return {'stopped': r.name if r else None}

    @app.post('/api/model/load')
    def model_load(req: LoadReq):
        d = _tm().run_dir(req.run)                          # only runs under RFCAV_RUNS_ROOT (pickle!)
        if not os.path.isdir(d):
            raise HTTPException(404, 'no such run')
        try:
            return service.load(d)
        except Exception as e:                               # noqa: BLE001
            raise HTTPException(422, f"could not load {req.run}: {e}") from None

    dist = os.environ.get('RFCAV_WEB_DIST', os.path.join(ROOT, 'web', 'dist'))
    if os.path.isdir(dist):
        app.mount('/assets', StaticFiles(directory=os.path.join(dist, 'assets')), name='assets')

        @app.get('/{path:path}', include_in_schema=False)
        def spa(path: str):
            f = os.path.join(dist, path)
            if path and os.path.isfile(f) and os.path.realpath(f).startswith(os.path.realpath(dist)):
                return FileResponse(f)
            return FileResponse(os.path.join(dist, 'index.html'))

    return app


def __getattr__(name):           # `uvicorn src.service.api:app` builds the app lazily on first access
    if name == 'app':
        globals()['app'] = create_app()
        return globals()['app']
    raise AttributeError(name)
