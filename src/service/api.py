"""Web API of the cavity eigenmode predictor (FastAPI). docs/26_WEB_UI.md.

    RFCAV_CHECKPOINT=runs/large uvicorn src.service.api:app --port 8000
    (no RFCAV_CHECKPOINT: an UNTRAINED demo model — for UI development only)

Endpoints (JSON; large arrays as base64 little-endian float32 / uint32 / float16, key suffix _b64):
    GET  /api/info                               model, families, limits
    POST /api/geometries            (multipart)  upload CAD / mesh → checks + display surface (+ can_predict)
    POST /api/geometries/sample     {family,id}  a generator geometry → checks + display surface (+ can_predict)
    POST /api/predictions           {geometry_id}           field3d model: mode table + figures of merit
    GET  /api/predictions/{id}/plane?mode&axis&pos&res      cut plane E/H of mode k (exact p2 evaluation)
    GET  /api/predictions/{id}/export             everything as one JSON file
    GET  /api/geometries/{id}/features            model input features (names, ranges)
    GET  /api/geometries/{id}/feature_plane|feature_surface?name&…   one feature on a cut / the wall
    GET  /api/predictions/{id}/volume?mode&n         mode on a regular 3D grid (exact p2 evaluation)
    GET  /api/predictions/{id}/fields               all modes at the P2 nodes + the sub-tet mesh + the wall values
                                                    (the workspace, client side; src/viz/hcurl_display.py)
    GET  /api/datasets                            registered datasets (RFCAV_DATA, src/service/datasets.py)
    GET  /api/datasets/{d}/items | /stats         geometry table (filter, sort, page) / overview
    POST /api/datasets/{d}/items/{id}/open        geometry + FE solution (+ labels, parameters)
    POST /api/predictions {geometry_id, compare_to}  + model-vs-FE rows and a difference solution
    GET  /api/jobs/config · POST /api/jobs/generate · GET /api/jobs[/{id}] · POST /api/jobs/{id}/cancel
    GET  /api/jobs/plan?max_ndof&threads&workers   field labels: workers × threads for this machine
    GET  /api/train/config · GET /api/train/runs[/{name}] · POST /api/train/start · /runs/{name}/resume
    POST /api/train/stop · POST /api/model/load {run}   training from the UI (RFCAV_RUNS_ROOT)
    GET  /api/truba/config · POST /api/truba/command {form}   one-line TRUBA command (src/service/truba.py)
The built frontend (web/dist) is served at / when present.

The fields are the HCurl(p2) fields themselves, drawn on the P2 nodes / sub-tets of the curved mesh and its curved
wall (docs/26 "Display pipeline"); no Whitney-N0 projection is involved.
Prediction runs the field3d model on the geometry's curved mesh (src/service/model.py): only geometries with a
CAD solid (CAD uploads, generator families except freeform, field-label dataset items) and at most
Limits.max_predict_tets curved tets can be predicted — the geometry response says so (can_predict,
predict_blocker) and POST /api/predictions answers 422 with that reason otherwise.
"""
import base64
import collections
import hashlib
import hmac
import logging
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

from src.data.mesh_features import FEATURE_NAMES_3D
from src.service import geometry as G
from src.service import training as T
from src.service import truba as TR

VERSION = '0.1.0'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_FAMILIES = ('elliptical', 'reentrant', 'pillbox_pipes', 'ridged_box', 'composite', 'freeform',
                  'hwr', 'spoke', 'dtl')
OOD_FAMILIES = ('box', 'coax_qw', 'pillbox_port', 'elliptical_long', 'junction')
# Random example ids (POST /api/geometries/sample without an id): beyond every dataset id.  Dataset ids are
# block·2^19 + i with block ≤ 31 (cluster/truba/families.tsv: train 1–9, ood 20–24, new families ≤ 31), i.e.
# all below 32·2^19 = 2^24; the random ids use [2^24, 2^25) (blocks 32–63, never assigned).  An explicit id
# is used as given (deterministic geometry).
RANDOM_SAMPLE_IDS = (2 ** 24, 2 ** 25)
UPLOAD_AXIS_WARNING = ('beam axis assumed: z through the origin; L_acc = axis chord incl. pipes; R/Q, R_sh, T and '
                       'peaks/Eacc depend on it')
TRAIN_MODEL_TETS = 10_000       # field_labels.DEFAULTS['model_max_elements']: the training meshes
BIG_MESH_WARNING = ('the curved mesh has {} tets, more than the <= {} of the training meshes: the model was '
                    'not trained on meshes this fine, accuracy is not validated')
SHARP_EDGE_WARNING = ('sharp edges ({} re-entrant): Epk/Eacc and Bpk/Eacc are not converged and are not reported')
log = logging.getLogger('rfcav.api')


def _env(name, default, cast=float):
    v = os.environ.get(name)
    return cast(v) if v not in (None, '') else default


class Limits:
    max_upload_mb = _env('RFCAV_MAX_UPLOAD_MB', 25.0)
    mesh_timeout_s = _env('RFCAV_MESH_TIMEOUT', 120.0)
    mesh_mem_mb = _env('RFCAV_MESH_MEM_MB', 4096, int)
    max_tets = _env('RFCAV_MAX_TETS', 400_000, int)
    # curved tets the field model predicts on (CPU: ~3 GB, ~15 s at 10k tets); the training model meshes have
    # ≤ 10k tets (field_labels model_max_elements), so 15k keeps a margin without running far outside them
    max_predict_tets = _env('RFCAV_MAX_PREDICT_TETS', 30_000, int)
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
    id: int | None = Field(default=None, ge=0, lt=RANDOM_SAMPLE_IDS[1])   # a random id can be re-entered
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
    from fastapi.middleware.gzip import GZipMiddleware
    app.add_middleware(GZipMiddleware, minimum_size=4096, compresslevel=3)   # tunnels (Colab, phone): fewer bytes
    app.state.service = service
    _access_token(app, os.environ.get('RFCAV_TOKEN') or None)

    def _blocker(g):
        """Why geometry record g cannot be predicted (None: it can)."""
        if not g['check']['ok']:
            return 'geometry failed the checks: ' + '; '.join(g['check']['problems'])
        if g.get('blocker'):
            return g['blocker']
        if g.get('mesh_src') is None:
            return 'no curved mesh of a CAD solid for this geometry'
        n = int(g.get('n_tets_curved') or len(g['tets']))
        if n > lim.max_predict_tets:
            return (f"the curved mesh has {n} tets, above the prediction budget of {lim.max_predict_tets} "
                    "(RFCAV_MAX_PREDICT_TETS): try a coarser mesh size (the curvature of the solid can keep "
                    "the curved mesh fine)")
        return None

    def _geometry_response(nodes, tets, source, extra=None, prepared=None, field=None):
        """Store a geometry; field: {mesh_src, blocker, n_tets, geom_params} of its curved mesh (or why none)."""
        if len(tets) > lim.max_tets:
            raise HTTPException(413, f"mesh has {len(tets)} tets (limit {lim.max_tets}); use a coarser mesh size")
        field = field or {}
        curved = field.get('curved')                         # the curved wall lattice (hcurl_display.curved_display)
        chk = G.check(nodes, tets, curved)
        rec = {'nodes': nodes, 'tets': tets, 'source': source,
               'family': (extra or {}).get('family', ''), 'params': (extra or {}).get('params'), 'check': chk,
               'mesh_src': field.get('mesh_src'), 'blocker': field.get('blocker'),
               'n_tets_curved': field.get('n_tets'), 'geom_params': field.get('geom_params')}
        if curved is not None:                               # the curved wall; vid3 / w3 carry vertex data onto it
            pts, tri = np.asarray(curved['points']) * 1e3, curved['triangles']
            rec.update(surf_vid3=curved['vid3'], surf_w3=curved['w3'])
        else:                                                # no curved mesh (mesh upload, freeform): linear facets
            rec['vid'], tri = G.surface(nodes, tets)
            pts = np.asarray(nodes)[rec['vid']] * 1e3
        rec['n_surface'] = int(len(pts))
        if prepared:
            rec.update(prepared)
        gid = geoms.put(rec)
        blocker = _blocker(rec)
        return {'id': gid, 'source': source, 'check': chk, **(extra or {}),
                'can_predict': blocker is None, 'predict_blocker': blocker,
                'surface': {'n_points': int(len(pts)), 'n_triangles': int(len(tri)),
                            'points_b64': b64(pts), 'triangles_b64': b64(tri, np.uint32)}}

    @app.get('/api/info')
    def info():
        return {'version': VERSION, 'model': service.info(),
                'families': {'train': list(TRAIN_FAMILIES), 'ood': list(OOD_FAMILIES)},
                'limits': {'max_upload_mb': lim.max_upload_mb, 'max_tets': lim.max_tets,
                           'max_predict_tets': lim.max_predict_tets,
                           'cad': list(G.CAD_EXT), 'mesh': list(G.MESH_EXT)},
                'conventions': {'U_J': 1.0, 'wall': 'copper, sigma = 5.8e7 S/m', 'beta': 1.0,
                                'R_over_Q': 'linac: V^2/(omega U)',
                                'beam_axis': 'src.qoi.figures.beam_axis: x = y = 0 along z; hwr, ridged_box, box per '
                                             'family; CAD uploads: z through the origin of the file',
                                'L_acc': 'cavity length along the beam, pipes excluded (acceleration_length); CAD '
                                         'uploads and families without one: the axis chord inside the cavity, pipes '
                                         'included'}}

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
                m = G.run_isolated(G.mesh_upload, path, unit, mesh_size, n_cells, None,
                                   timeout=lim.mesh_timeout_s, mem_mb=lim.mesh_mem_mb)
            except (RuntimeError, TimeoutError) as e:
                raise HTTPException(422, f"could not mesh the file: {e}") from None
        # unit_used: 'file' for STEP / IGES (the unit the file declares; `unit` ignored), else `unit`
        n_sharp = m.get('n_sharp_edges')
        return _geometry_response(m['nodes'], m['tets'], {'kind': 'upload', 'name': os.path.basename(file.filename or ''),
                                                          'unit': unit, 'unit_used': m.get('unit_used', unit),
                                                          'mesh_size': mesh_size, 'n_cells': n_cells,
                                                          'n_sharp_edges': n_sharp},
                                  field=dict(m, geom_params=None if n_sharp is None else {'n_sharp_edges': float(n_sharp)}))

    @app.post('/api/geometries/sample')
    def sample(req: SampleReq):
        if req.family not in TRAIN_FAMILIES + OOD_FAMILIES:
            raise HTTPException(422, f"unknown family {req.family!r}")
        s_id = req.id if req.id is not None else random.randrange(*RANDOM_SAMPLE_IDS)   # beyond all dataset ids
        try:
            m = G.run_isolated(G.mesh_family, req.family, s_id, req.mesh_size,
                               timeout=lim.mesh_timeout_s, mem_mb=lim.mesh_mem_mb)
        except (RuntimeError, TimeoutError) as e:
            raise HTTPException(422, f"could not build the geometry: {e}") from None
        return _geometry_response(m['nodes'], m['tets'], {'kind': 'sample', 'family': req.family, 'id': s_id,
                                                          'mesh_size': req.mesh_size, 'ood': req.family in OOD_FAMILIES},
                                  {'family': req.family, 'params': m['params']}, field=dict(m, geom_params=m['params']))

    def _prepared(g):
        """Display geometry (normalised linear mesh + input features) of a stored geometry, computed once (only the
        feature views need it; the field display does not)."""
        if '_geom' not in g:
            g['_geom'] = service.geometry(g['nodes'], g['tets'], g['family'], g.get('geom_params') or g.get('params'))
        return g['_geom']

    def _summary(sid, sol, extra=None):
        return {'id': sid, 'source': sol['source'], 'n_dof': sol['n_dof'],
                'modes': sol['modes'], 'axis_length_mm': sol['axis_length_mm'], **(extra or {})}

    @app.post('/api/predictions')
    def predict(req: PredictReq):
        g = geoms.get(req.geometry_id)
        blocker = _blocker(g)
        if blocker:
            raise HTTPException(422, blocker)
        t0 = time.perf_counter()
        try:
            p = service.predict(g['mesh_src'], g['family'], g.get('geom_params'), key=req.geometry_id)
        except Exception as e:                   # noqa: BLE001 — CG non-convergence, netgen …
            log.exception('prediction of geometry %s failed', req.geometry_id)
            raise HTTPException(422, f"prediction failed ({type(e).__name__}: {str(e)[:300]})") from None
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
        if p['n_tets'] > TRAIN_MODEL_TETS:
            warnings.append(BIG_MESH_WARNING.format(p['n_tets'], TRAIN_MODEL_TETS))
        if g['source'].get('kind') == 'upload':
            warnings.append(UPLOAD_AXIS_WARNING)
        if float((g.get('geom_params') or {}).get('n_sharp_edges') or 0) > 0:
            warnings.append(SHARP_EDGE_WARNING.format(int(g['geom_params']['n_sharp_edges'])))
        out = _summary(pid, p, {'geometry_id': req.geometry_id, 'time_s': p['time_s'], 'warnings': warnings,
                                'model': service.info(), 'qoi_source': p['qoi_source'], 'n_dof': p['n_dof'],
                                'n_tets': p['n_tets']})
        if req.compare_to:
            from src.service.model import compare
            truth = preds.get(req.compare_to)
            if truth.get('_geometry_id') != req.geometry_id:
                raise HTTPException(422, 'compare_to must be the FE solution of the same geometry')
            try:
                rows, diff = compare(p, truth)
            except Exception as e:               # noqa: BLE001
                log.exception('comparison of geometry %s failed', req.geometry_id)
                raise HTTPException(422, f"comparison failed ({type(e).__name__}: {str(e)[:300]})") from None
            diff['_geometry_id'] = req.geometry_id
            out['comparison'] = {'rows': rows, 'diff_id': preds.put(diff), 'truth_id': req.compare_to}
        return out

    @app.get('/api/predictions/{pid}/fields')
    def fields(pid: str):
        """Every mode of a solution on the display mesh in one response: the workspace cuts planes, fills the 3D
        volume and applies modes, phase and E/H in the browser (no request per change).
        nodes_b64 [n_nodes, 3] float32 mm (the P2 nodes of the curved mesh), tets_b64 [n_cells, 4] uint32 (8 linear
        sub-tets per tet on those nodes), E_b64 / H_b64 [K, n_nodes, 3] float16 (each mode divided by its E_max /
        H_max), surf_E_b64 / surf_H_b64 [K, n_surface] float16 (|E|, |H| on the geometry's curved wall points, same
        divisors; E_max / H_max cover nodes and wall), axis: beam-axis E_z per mode (null outside the cavity)."""
        p = preds.get(pid)
        g = geoms.get(p['_geometry_id'])
        try:
            r = service.fields(p)
        except Exception as e:                   # noqa: BLE001 — mesh reload, NGSolve version, …
            log.exception('fields of prediction %s failed', pid)
            raise HTTPException(422, f"display failed ({type(e).__name__}: {str(e)[:300]})") from None
        K, ns = r['E'].shape[0], r['sE'].shape[1]
        if ns != g['n_surface']:
            raise HTTPException(500, f"the wall lattice has {ns} points, the geometry's surface {g['n_surface']}")
        e_max, h_max = r['E_max'], r['H_max']
        return {'n_modes': int(K), 'n_nodes': int(r['E'].shape[1]), 'n_cells': int(len(r['tets'])),
                'n_tets': int(p.get('n_tets') or 0), 'n_surface': int(ns),
                'f_GHz': [float(v) for v in p['_f']], 'E_max': e_max.tolist(), 'H_max': h_max.tolist(),
                'nodes_b64': b64(r['x_mm']), 'tets_b64': b64(r['tets'], np.uint32),
                'E_b64': b64(r['E'] / e_max[:, None, None], np.float16),
                'H_b64': b64(r['H'] / h_max[:, None, None], np.float16),
                'surf_E_b64': b64(r['sE'] / e_max[:, None], np.float16),
                'surf_H_b64': b64(r['sH'] / h_max[:, None], np.float16),
                'axis': {'z_mm': [float(v) for v in r['axis'][0]],
                         'Ez': [[_clean(v) for v in row] for row in r['axis'][1]]}}

    @app.get('/api/predictions/{pid}/plane')
    def plane(pid: str, mode: int = 0, axis: str = 'y', pos: float | None = None, res: int = 121):
        p = preds.get(pid)
        if axis not in ('x', 'y', 'z') or not 16 <= res <= 241 or not 0 <= mode < len(p['_f']):
            raise HTTPException(422, 'axis in x|y|z, res in [16, 241], valid mode')
        pts, tri, E, H = service.plane(p, mode, axis, pos, res)
        return {'mode': mode, 'axis': axis, 'n_points': int(len(pts)), 'n_triangles': int(len(tri)),
                'points_b64': b64(pts), 'triangles_b64': b64(tri, np.uint32), 'E_b64': b64(E), 'H_b64': b64(H)}

    @app.get('/api/predictions/{pid}/volume')
    def volume(pid: str, mode: int = 0, n: int = 80_000):
        """Mode on a regular 3D grid (x fastest, vtkImageData order; 0 outside): the 3D field view."""
        p = preds.get(pid)
        if not 5_000 <= n <= 300_000 or not 0 <= mode < len(p['_f']):
            raise HTTPException(422, 'n in [5000, 300000], valid mode')
        v = service.volume(p, mode, n)
        return {'mode': mode, 'dims': v['dims'], 'origin_mm': v['origin_mm'], 'spacing_mm': v['spacing_mm'],
                'n_points': int(np.prod(v['dims'])), 'n_inside': int(v['inside'].sum()),
                'inside_b64': base64.b64encode(np.ascontiguousarray(v['inside'], np.uint8).tobytes()).decode(),
                'E_b64': b64(v['E']), 'H_b64': b64(v['H'])}

    @app.get('/api/predictions/{pid}/export')
    def export(pid: str):
        p = preds.get(pid)
        g = geoms.get(p['_geometry_id'])
        body = {'version': VERSION, 'model': service.info(), 'geometry': {'source': g['source'], 'check': g['check']},
                'source': p['source'], 'modes': p['modes'], 'time_s': p.get('time_s'),
                'qoi_source': p.get('qoi_source'),
                'conventions': info()['conventions']}
        return JSONResponse(body, headers={'Content-Disposition': f'attachment; filename="prediction_{pid[:8]}.json"'})

    # ── model input features ─────────────────────────────────────────
    def _feature(g, name):
        geom = _prepared(g)
        names = list(g.get('feature_names') or FEATURE_NAMES_3D)
        if name not in names:
            raise HTTPException(404, f"unknown feature {name!r}")
        return geom, np.asarray(geom['Input_funcs'], float)[:, names.index(name)]

    @app.get('/api/geometries/{gid}/features')
    def features(gid: str):
        g = geoms.get(gid)
        geom = _prepared(g)
        names = list(g.get('feature_names') or FEATURE_NAMES_3D)
        F = np.asarray(geom['Input_funcs'], float)
        return {'names': names, 'min': F.min(0).tolist(), 'max': F.max(0).tolist(),
                'n_vertices': int(len(F)), 'n_tets': int(len(geom['tets'])),
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
        if 'surf_vid3' in g:                                  # the curved wall lattice: P1 data of the face's vertices
            v = np.einsum('pk,pk->p', v[g['surf_vid3']], g['surf_w3'])
        else:
            v = v[g['vid']]
        return {'name': name, 'values_b64': b64(v)}

    @app.get('/api/geometries/{gid}/feature_view')
    def feature_view_ep(gid: str, kind: str, channel: str | None = None, axis: str = 'y',
                        pos: float | None = None, res: int = 121):
        from src.service.features import KINDS, feature_view
        if kind not in KINDS or axis not in ('x', 'y', 'z') or not 16 <= res <= 241:
            raise HTTPException(422, f"kind in {KINDS}, axis in x|y|z, res in [16, 241]")
        g = geoms.get(gid)
        geom = _prepared(g)
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
        return d.rows()

    @app.get('/api/datasets')
    def datasets_list():
        return [{'id': i, 'name': d.name, 'kind': d.kind} for i, d in ds_ids.items()]

    @app.get('/api/datasets/{did}/items')
    def dataset_items(did: str, family: str | None = None, offset: int = 0, limit: int = 50,
                      sort: str = 'id', desc: bool = False):
        rows = _rows(_ds(did))
        if family:
            rows = [r for r in rows if r['family'] == family]
        keyf = {'id': lambda r: r['id'], 'family': lambda r: r['family'], 'n_dof': lambda r: r['n_dof'],
                'n_nodes': lambda r: r['n_nodes'], 'betti1': lambda r: r['betti1'],
                'f0': lambda r: r['f_GHz'][0] if r['f_GHz'] else np.inf}.get(sort)
        if keyf is None:
            raise HTTPException(422, 'sort by id | family | n_dof | n_nodes | betti1 | f0')
        rows = sorted(rows, key=keyf, reverse=desc)
        limit = max(1, min(int(limit), 500))
        return {'total': len(rows), 'offset': offset, 'items': rows[offset:offset + limit]}

    @app.get('/api/datasets/{did}/stats')
    def dataset_stats(did: str):
        rows = _rows(_ds(did))
        fam = collections.defaultdict(lambda: {'count': 0, 'f0': [], 'n_dof': []})
        for r in rows:
            e = fam[r['family']]
            e['count'] += 1
            e['n_dof'].append(r['n_dof'])
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
        field = {'mesh_src': r.get('mesh_src'), 'n_tets': r.get('n_tets_model'), 'geom_params': r['geom_params'],
                 'curved': r['curved'], 'blocker': None if r.get('mesh_src') else 'no curved mesh is stored for this item'}
        resp = _geometry_response(r['nodes'], r['tets'],
                                  {'kind': 'dataset', 'dataset': d.name, 'family': r['family'], 'id': r['id'],
                                   'file': r['source_file']},
                                  {'family': r['family'], 'params': r['params']}, field=field)
        truth = None
        if r['U'].shape[1]:
            sol = service.fe_solution(r, resp['id'])     # the stored p3 labels are the figures of merit shown
            sol['_geometry_id'] = resp['id']
            truth = _summary(preds.put(sol), sol, {'f_next': r['f_next'], 'labels': r['qoi_labels'],
                                                   'qoi_source': 'p3_labels' if r['qoi_labels'] else 'fe'})
        resp.update(truth=truth)
        return resp

    @app.post('/api/datasets/rescan')
    def datasets_rescan():
        rescan()
        return datasets_list()

    # ── dataset generation (background jobs; RFCAV_GEN_ROOT, e.g. a Google Drive folder) ──
    from src.service import jobs as J
    blocks = J.family_blocks()
    FIELD_PARAMS = {'threads', 'label_order', 'model_order', 'min_fillet', 'maxh_factor', 'field_max_elements',
                    'adapt', 'tol_f', 'tol_q', 'model_max_elements', 'max_ndof', 'round_edges', 'heal',
                    'sample_timeout'}
    jm = J.JobManager(gen_root, on_done=lambda job: rescan()) if gen_root else None
    max_gen = _env('RFCAV_GEN_MAX', 20000, int)

    class GenReq(BaseModel):
        families: list[str]
        labels: str = Field(default='field', pattern='^field$')        # field labels only (the field3d data)
        n_total: int = Field(default=100, ge=1)
        mesh_size: float = Field(default=0.10, ge=Limits.min_mesh_size, le=0.5)
        n_modes: int = Field(default=10, ge=1, le=30)
        sampling: str = Field(default='sobol', pattern='^(sobol|random)$')
        deform_prob: float = Field(default=0.5, ge=0.0, le=1.0)
        deform_max: float = Field(default=0.5, ge=0.0, le=0.9)
        seed: int = Field(default=0, ge=0)
        shard_size: int = Field(default=256, ge=4, le=4096)
        workers: int | None = Field(default=None, ge=1, le=256)
        tag: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$')
        # field labels (docs/29–30)
        threads: int | None = Field(default=None, ge=1, le=256)     # NGSolve threads per worker (None: plan)
        label_order: int = Field(default=3, ge=2, le=5)
        model_order: int = Field(default=2, ge=1, le=4)
        min_fillet: float = Field(default=0.05, ge=0.0, le=0.3)
        maxh_factor: float = Field(default=3.0, ge=1.0, le=10.0)
        field_max_elements: int = Field(default=30000, ge=1000, le=500000)
        adapt: bool = True
        tol_f: float = Field(default=1e-6, ge=1e-9, le=1e-2)
        tol_q: float = Field(default=1e-3, ge=1e-6, le=0.2)
        model_max_elements: int = Field(default=10000, ge=100, le=500000)
        max_ndof: int | None = Field(default=None, ge=10000, le=20_000_000)  # None: the machine's default
        round_edges: bool = True
        heal: bool = True
        sample_timeout: int = Field(default=7200, ge=60, le=86400)

    def _jm():
        if jm is None:
            raise HTTPException(503, 'dataset generation is off: start the server with RFCAV_GEN_ROOT=<folder> '
                                     '(e.g. a Google Drive folder)')
        return jm

    host = J.machine()

    def _default_ndof():
        """Largest standard label budget at which one field worker fits this machine's memory."""
        fits = [n for n in (900_000, 600_000, 300_000, 150_000)
                if not host['mem_gb'] or J.field_mem_gb(n) <= 0.85 * host['mem_gb']]
        return fits[0] if fits else 60_000

    @app.get('/api/jobs/config')
    def jobs_config():
        nd = _default_ndof()
        return {'enabled': jm is not None, 'gen_root': gen_root,
                'cpu_count': host['cpus'], 'mem_gb': host['mem_gb'], 'max_total': max_gen,
                'families': {'train': [f for f in TRAIN_FAMILIES], 'ood': [f for f in OOD_FAMILIES]},
                'blocks': {f: blocks.get(f) for f in TRAIN_FAMILIES + OOD_FAMILIES},
                'tag_example': J.field_tag_of(0.10, 10),
                'field': {'defaults': {k: v.default for k, v in GenReq.model_fields.items()
                                       if k in FIELD_PARAMS} | {'max_ndof': nd},
                          'plan': J.field_plan(host['cpus'], host['mem_gb'], nd),
                          'core_s': J.FIELD_CORE_S, 'adapt_factor': J.FIELD_ADAPT_FACTOR,
                          'no_cad': ['freeform']}}

    @app.get('/api/jobs/plan')
    def jobs_plan(max_ndof: int | None = None, threads: int | None = None, workers: int | None = None):
        """Field labels: workers × threads and the memory they need on this machine (J.field_plan)."""
        for name, v, lo in (('max_ndof', max_ndof, 10000), ('threads', threads, 1), ('workers', workers, 1)):
            if v is not None and v < lo:
                raise HTTPException(422, f"{name} ≥ {lo}")
        return J.field_plan(host['cpus'], host['mem_gb'], max_ndof or _default_ndof(), threads, workers)

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
        if 'freeform' in req.families:
            raise HTTPException(422, 'freeform has no CAD solid: field labels need one (docs/29)')
        if req.model_order >= req.label_order:
            raise HTTPException(422, 'model_order must be below label_order (nested spaces)')
        nd = req.max_ndof or _default_ndof()
        plan = J.field_plan(host['cpus'], host['mem_gb'], nd, req.threads, req.workers)
        tag = req.tag or J.field_tag_of(req.mesh_size, req.n_modes, req.label_order, req.model_order,
                                        req.min_fillet, req.adapt, req.round_edges)
        out = []
        for f in req.families:
            p = req.model_dump(exclude={'families', 'tag', 'workers', 'threads', 'max_ndof'})
            p.update(family=f, block=blocks[f], tag=tag, group='ood' if f in OOD_FAMILIES else 'train',
                     workers=plan['workers'], threads=plan['threads'], max_ndof=nd)
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
        batch_size: int = Field(default=2, ge=1, le=256)                  # configs/field_3d.yaml
        lr: float = Field(default=2e-4, gt=0, le=1.0)
        n_modes: int = Field(default=6, ge=1, le=30)
        patience: int = Field(default=40, ge=1, le=10000)
        num_workers: int = Field(default=2, ge=0, le=64)                   # loader workers (operator assembly)

    class ResumeReq(BaseModel):
        epochs: int | None = Field(default=None, ge=1, le=10000)

    class LoadReq(BaseModel):
        run: str = Field(pattern=T.NAME_RE)

    def _tm():
        if tm is None:
            raise HTTPException(503, 'training is off: start the server with RFCAV_RUNS_ROOT=<folder> '
                                     '(e.g. <Drive>/training_logs)')
        return tm

    def _train_data(ids):
        """{'h5': files} of the chosen field-label datasets (the field3d model's data) + their names."""
        if not ids:
            raise HTTPException(422, 'choose at least one dataset')
        ds = [_ds(i) for i in ids]
        return {'h5': sorted({f for d in ds for f in d.files})}, [d.name for d in ds]

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
                'model': service.info(), 'model_type': 'field3d', 'config': 'configs/field_3d.yaml',
                'dataset_kinds': ['field'],
                'defaults': {k: v.default for k, v in TrainReq.model_fields.items() if k not in ('name', 'datasets')}}

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
        if not (data or {}).get('h5'):
            raise HTTPException(422, f"{name}: train_params.json lists no field-label H5 files; it cannot be resumed")
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
