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
The built frontend (web/dist) is served at / when present.
"""
import base64
import collections
import os
import random
import tempfile
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.service import geometry as G

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


def create_app(service=None):
    """FastAPI app around a ModelService (built from RFCAV_CHECKPOINT / RFCAV_DEVICE when None)."""
    if service is None:
        from src.service.model import ModelService
        service = ModelService(os.environ.get('RFCAV_CHECKPOINT') or None, os.environ.get('RFCAV_DEVICE') or None)
    lim = Limits()
    geoms, preds = Store(lim.max_items, lim.ttl_s), Store(lim.max_items, lim.ttl_s)
    app = FastAPI(title='RF Cavity Neural Solver', version=VERSION)
    app.state.service = service

    def _geometry_response(nodes, tets, source, extra=None):
        if len(tets) > lim.max_tets:
            raise HTTPException(413, f"mesh has {len(tets)} tets (limit {lim.max_tets}); use a coarser mesh size")
        chk = G.check(nodes, tets, service.field)
        vid, tri = G.surface(nodes, tets)
        gid = geoms.put({'nodes': nodes, 'tets': tets, 'vid': vid, 'tri': tri, 'source': source,
                         'family': (extra or {}).get('family', ''), 'check': chk})
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
                nodes, tets = G.run_isolated(G.mesh_upload, path, unit, mesh_size, n_cells, None, service.field,
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
            nodes, tets, params = G.run_isolated(G.mesh_family, req.family, s_id, service.field, req.mesh_size,
                                                 timeout=lim.mesh_timeout_s, mem_mb=lim.mesh_mem_mb)
        except (RuntimeError, TimeoutError) as e:
            raise HTTPException(422, f"could not build the geometry: {e}") from None
        return _geometry_response(nodes, tets, {'kind': 'sample', 'family': req.family, 'id': s_id,
                                                'mesh_size': req.mesh_size, 'ood': req.family in OOD_FAMILIES},
                                  {'family': req.family, 'params': params})

    @app.post('/api/predictions')
    def predict(req: PredictReq):
        g = geoms.get(req.geometry_id)
        if not g['check']['ok']:
            raise HTTPException(422, 'geometry failed the checks: ' + '; '.join(g['check']['problems']))
        t0 = time.perf_counter()
        p = service.predict(g['nodes'], g['tets'], g['family'])
        p['time_s']['total'] = time.perf_counter() - t0
        p['_geometry_id'] = req.geometry_id
        pid = preds.put(p)
        warnings = []
        if service.untrained:
            warnings.append('untrained demo model: the numbers are meaningless (UI development mode)')
        if p['axis_length_mm'] <= 0:
            warnings.append('the beam axis (x = y = 0 along z) does not cross the cavity: R/Q, R_sh, T, '
                            'Epk/Eacc and Bpk/Eacc are undefined')
        if g['source'].get('ood'):
            warnings.append('out-of-distribution family: verify with a full-wave solver')
        return {'id': pid, 'geometry_id': req.geometry_id, 'field': p['field'], 'n_edges': p['n_edges'],
                'modes': p['modes'], 'axis_length_mm': p['axis_length_mm'], 'time_s': p['time_s'],
                'warnings': warnings, 'model': service.info()}

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
                'field': p['field'], 'modes': p['modes'], 'time_s': p['time_s'],
                'conventions': info()['conventions']}
        return JSONResponse(body, headers={'Content-Disposition': f'attachment; filename="prediction_{pid[:8]}.json"'})

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
