"""Cavity figures of merit (Q0, G, R/Q, R_sh, T, E_pk/E_acc, B_pk/E_acc) from N0 eigenmode DOFs.

numpy reference: src/qoi/operators.py; closed-form references: src/qoi/analytic.py;
differentiable torch version: src/qoi/torch_qoi.py (import it explicitly).  Contract: docs/24 §0.
"""
from src.qoi.operators import (QOI_KEYS, QOI_LABELS, SIGMA_CU, build_qoi_operators, cavity_qoi,
                               qoi_from_dofs, surface_resistance)

__all__ = ["QOI_KEYS", "QOI_LABELS", "SIGMA_CU", "build_qoi_operators", "cavity_qoi", "qoi_from_dofs",
           "surface_resistance"]
