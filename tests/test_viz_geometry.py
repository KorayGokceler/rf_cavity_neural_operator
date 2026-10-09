"""src/viz/geometry.py: load generator H5 geometries and render the gallery."""
import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("gmsh")
pytest.importorskip("skfem")

import src.data_gen.dataset_generator_3d as gen  # noqa: E402
from src.viz.geometry import gallery, load_h5  # noqa: E402


def test_load_and_gallery(tmp_path):
    h5 = tmp_path / "g.h5"
    old = gen.ARGS
    try:
        gen.main(["--n_total", "3", "--n_workers", "2", "--mesh_size", "0.25", "--n_eigen_modes", "2",
                  "--h5_filename", str(h5), "--seed", "1"])
    finally:
        gen.ARGS = old
    geoms = load_h5(str(h5), n=2)
    assert len(geoms) == 2 and len({g["shape_type"] for g in geoms}) >= 1
    for g in geoms:                                   # closed surface: every edge in exactly 2 faces
        e = np.sort(g["faces"][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        assert (cnt == 2).all()
    assert [g["id"] for g in load_h5(str(h5), ids=[2])] == [2]
    fig = gallery(geoms)
    assert len(fig.axes) == 4
    fig.savefig(tmp_path / "g.png")
