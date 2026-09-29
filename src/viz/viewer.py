"""CST-style interactive mode viewer (Colab / Jupyter, ipywidgets + matplotlib).

One geometry (a src/viz/predict.py dict): true | predicted | |error| on an
axis-aligned cut, with a phase slider 0–180° and a cut-position slider [mm].
Fields are nodal-averaged (nedelec.vertex_field); colour limits are fixed per
(mode, field, component) by the true field over the whole volume, so moving
the phase / the cut shows the real variation.

The primary field is the one the DOFs are (pred['field']: 'H' default, 'E'
for E-formulation PKLs); the other one is the curl of the primary.  A
lossless eigenmode is a standing wave, E and H 90° apart:
    H primary:  H(t) = H·cos φ,  E(t) ∝ +curl H·sin φ   (Ampère, 1/(jωε))
    E primary:  E(t) = E·cos φ,  H(t) ∝ −curl E·sin φ   (Faraday, −1/(jωμ))
So the pattern does not travel; 0° = primary maximal / secondary zero,
90° = secondary maximal / primary zero, 180° = primary with the opposite sign.
(Amplitude constants are dropped: each field has its own colour limit.)

    from src.viz.predict import load, predict
    from src.viz.viewer import ModeViewer
    lm, ds = load(ckpt_dir, pkl, 'test')
    ModeViewer(predict(lm, ds, 0)).widget()
"""
import matplotlib.pyplot as plt
import numpy as np

from src.viz.nedelec import interp_vertex, locate, plane_grid, vertex_field

COMPONENTS = ('abs', 'x', 'y', 'z')
_AX = {'x': 0, 'y': 1, 'z': 2}


class ModeViewer:
    def __init__(self, pred, res=141):
        self.p, self.res = pred, res
        self.X = np.asarray(pred['X'], np.float64)
        self.tets = np.asarray(pred['tets'], np.int64)
        self.scale, self.center = float(pred['scale']), np.asarray(pred['center'], np.float64)
        self.K = pred['true'].shape[1]
        self.primary = str(pred.get('field', 'H') or 'H').upper()
        if self.primary not in ('H', 'E'):
            raise ValueError(f"pred['field'] must be 'H' or 'E', got {self.primary!r}")
        self.secondary = 'E' if self.primary == 'H' else 'H'
        self.fields = (self.primary, self.secondary)
        # secondary(t) ∝ sign · curl(primary) · sin φ  (see module docstring)
        self._sign = {self.primary: 1.0, self.secondary: 1.0 if self.primary == 'H' else -1.0}
        self.nodal = {(f, w): vertex_field(self.X, self.tets, pred['edges'], pred[w], curl=(f != self.primary))
                      for f in self.fields for w in ('true', 'pred')}      # [Nv,3,K]
        self._planes = {}

    # ── geometry ────────────────────────────────────────────────────────
    def mm(self, c, axis):
        return (c * self.scale + self.center[_AX[axis]]) * 1e3

    def range_mm(self, axis):
        c = self.X[:, _AX[axis]]
        return float(self.mm(c.min(), axis)), float(self.mm(c.max(), axis))

    def _plane(self, axis, pos_mm):
        key = (axis, round(pos_mm, 3))
        if key not in self._planes:
            off = (pos_mm * 1e-3 - self.center[_AX[axis]]) / self.scale
            pts, U, V, (iu, iv) = plane_grid(self.X, axis, off, self.res)
            tid, bary = locate(self.X, self.tets, pts)
            if len(self._planes) > 64:
                self._planes.clear()
            self._planes[key] = (tid, bary, U, V, iu, iv)
        return self._planes[key]

    # ── values ──────────────────────────────────────────────────────────
    @staticmethod
    def _comp(F, comp):
        return np.linalg.norm(F, axis=-1) if comp == 'abs' else F[..., _AX[comp]]

    def limit(self, k, field, comp):
        """Colour limit: max of the TRUE field over the volume (prediction saturates above)."""
        field = self._field(field)
        return float(np.nanmax(np.abs(self._comp(self.nodal[(field, 'true')][:, :, k], comp)))) or 1.0

    def _field(self, field):
        field = self.primary if field is None else str(field).upper()
        if field not in self.fields:
            raise ValueError(f"field must be one of {self.fields}, got {field!r}")
        return field

    def slice(self, k, field=None, comp='abs', axis='y', pos_mm=None, phase=0.0):
        """(U_mm, V_mm, true, pred, err) grids [res,res] at the given phase [deg];
        field None = the primary one."""
        field = self._field(field)
        if pos_mm is None:
            pos_mm = float(self.mm(0.0, axis))                       # volume centroid
        tid, bary, U, V, iu, iv = self._plane(axis, pos_mm)
        ph = np.radians(phase)
        t = np.cos(ph) if field == self.primary else self._sign[field] * np.sin(ph)
        F = {w: t * interp_vertex(self.tets, tid, bary, self.nodal[(field, w)][:, :, k])
             for w in ('true', 'pred')}
        shape = U.shape
        out = [self._comp(F[w], comp).reshape(shape) for w in ('true', 'pred')]
        err = np.linalg.norm(F['pred'] - F['true'], axis=-1).reshape(shape)
        lab = 'xyz'
        return (self.mm(U, lab[iu]), self.mm(V, lab[iv]), *out, err, lab[iu], lab[iv])

    # ── drawing ─────────────────────────────────────────────────────────
    def draw(self, k=0, field=None, comp='abs', axis='y', pos_mm=None, phase=0.0, fig=None):
        field = self._field(field)
        U, V, ft, fp, fe, lu, lv = self.slice(k, field, comp, axis, pos_mm, phase)
        lim = self.limit(k, field, comp)
        r = np.ptp(V) / max(np.ptp(U), 1e-12)
        if fig is None:
            fig = plt.figure(figsize=(14, float(np.clip(4.2 * r + 1.2, 2.6, 6.0))))
        fig.clear()
        axs = fig.subplots(1, 3)
        signed = comp != 'abs'
        cmap, vmin = ('RdBu_r', -lim) if signed else ('jet', 0.0)
        name = f"|{field}|" if comp == 'abs' else f"{field}_{comp}"
        if field != self.primary:
            name += f" (∝ curl {self.primary})"
        for ax, z, ttl, cm, lo in ((axs[0], ft, f"True {name}", cmap, vmin),
                                   (axs[1], fp, f"Predicted {name}", cmap, vmin),
                                   (axs[2], fe, f"|{field}_pred − {field}_true|", 'magma', 0.0)):
            im = ax.pcolormesh(U, V, np.ma.masked_invalid(z), cmap=cm, vmin=lo, vmax=lim, shading='auto')
            ax.set_aspect('equal')
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel(f"{lu} [mm]")
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        axs[0].set_ylabel(f"{lv} [mm]")
        p = self.p
        rl = p['rel_l2'][k]
        rl = 'n/a (split pair)' if p['split'][k] else ('n/a' if not np.isfinite(rl) else f"{rl:.3f}")
        pos = float(self.mm(0.0, axis)) if pos_mm is None else pos_mm
        fig.suptitle(f"geom {p['geom_id']} {p['shape_type']} · mode {k} · f_true {p['f_true'][k]:.4f} GHz"
                     f" · f_pred {p['f_pred'][k]:.4f} GHz · relL2 {rl} · {axis} = {pos:.1f} mm"
                     f" · phase {phase:.0f}°", fontsize=10)
        fig.tight_layout()
        return fig

    def widget(self):
        """ipywidgets UI: mode / field (primary first) / component / cut axis, cut
        position and phase sliders, ▶ plays the phase 0→180° in a loop."""
        import ipywidgets as w
        from IPython.display import display

        mode = w.Dropdown(options=list(range(self.K)), value=0, description='mode')
        field = w.ToggleButtons(options=list(self.fields), value=self.primary, description='field')
        comp = w.ToggleButtons(options=list(COMPONENTS), value='abs', description='comp.')
        axis = w.ToggleButtons(options=['x', 'y', 'z'], value='y', description='cut ⟂')
        lo, hi = self.range_mm('y')
        pos = w.FloatSlider(value=float(self.mm(0.0, 'y')), min=lo, max=hi, step=(hi - lo) / 100,
                            description='pos [mm]', continuous_update=False, readout_format='.1f',
                            layout=w.Layout(width='60%'))
        phase = w.IntSlider(value=0, min=0, max=180, step=5, description='phase [°]',
                            layout=w.Layout(width='60%'))
        play = w.Play(value=0, min=0, max=180, step=10, interval=250)
        w.jslink((play, 'value'), (phase, 'value'))

        def on_axis(ch):
            a, b = self.range_mm(ch['new'])
            pos.min, pos.max = min(a, pos.min), max(b, pos.max)      # widen first, then set
            pos.min, pos.max, pos.step = a, b, (b - a) / 100
            pos.value = float(self.mm(0.0, ch['new']))
        axis.observe(on_axis, names='value')

        out = w.Output()
        fig = plt.figure()
        plt.close(fig)

        def redraw(*_):
            with out:
                out.clear_output(wait=True)
                self.draw(mode.value, field.value, comp.value, axis.value, pos.value, phase.value, fig)
                display(fig)
        for c in (mode, field, comp, axis, pos, phase):
            c.observe(redraw, names='value')
        redraw()
        display(w.VBox([w.HBox([mode, field]), comp, axis, pos, w.HBox([phase, play]), out]))
