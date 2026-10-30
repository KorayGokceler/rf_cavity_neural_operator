"""Cavity figures of merit (Q0, G, R/Q, R_sh, T, E_pk/E_acc, B_pk/E_acc): conventions and rescaling in
src/qoi/figures.py, closed-form references in src/qoi/analytic.py.  The fields they are evaluated on are the
HCurl(p2) modes of src/data_gen/highorder.py (mode_qoi).  Contract: docs/24 §0.
"""
from src.qoi.figures import QOI_KEYS, QOI_LABELS, SIGMA_CU, surface_resistance

__all__ = ["QOI_KEYS", "QOI_LABELS", "SIGMA_CU", "surface_resistance"]
