import ast
import hashlib
import json
from pathlib import Path

import healpy as hp
import h5py
import numpy as np
import pytest


def test_numerical_helpers_are_unchanged_from_uploaded_baseline():
    import targ_ac_git.aux_snr_mf as aux
    source = Path(aux.__file__).read_text()
    expected = json.loads(Path(__file__).with_name("baseline_hashes.json").read_text())
    # compute_range also selects plotted curves, which this change updates.
    # Its saved priors and plotted curves are checked separately below.
    expected.pop("compute_range")
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    for name, digest in expected.items():
        assert hashlib.sha256(functions[name].encode()).hexdigest() == digest, name


def test_defaults_and_always_saved_isotropic_prior():
    from targ_ac_git.targ_range_snr_mf import build_parser, parse_iota_ranges
    from targ_ac_git.pipeline_utils_snr_mf import _build_offsets, _offset_label
    args = build_parser().parse_args(["--grb-name", "GRB200228291", "--output-dir", "out",
                                     "--t0", "2020-02-28T06:58:33.806", "--ra", "333.89292",
                                     "--dec=-42.944167"])
    assert (args.scan_start, args.scan_end, args.scan_step) == (0, 0, 32)
    assert (args.iota_min, args.iota_max, args.snr_threshold) == (0, 45, 9)
    assert [p["label"] for p in parse_iota_ranges()] == ["0-45", "0-90"]
    assert [p["label"] for p in parse_iota_ranges(10, 35)] == ["10-35", "0-90"]
    assert [p["label"] for p in parse_iota_ranges(0, 90)] == ["0-90"]
    assert _build_offsets(3.584, -100, 32) == [-100, -96, -64, -32, 0, 3.584]
    assert _offset_label(3.584) == "scan_t0p0003p584"


def test_all_three_range_contours_and_legend_patterns_are_retained(tmp_path, monkeypatch):
    from matplotlib.lines import Line2D
    from matplotlib.legend import Legend
    import targ_ac_git.aux_snr_mf as aux
    values = np.linspace(20, 520, 48)
    path = tmp_path / "range_map_bns_m1_1.4_m2_1.4.fits"
    hp.write_map(path, values, nest=True, dtype=float)
    monkeypatch.setattr(aux.pp, "close", lambda *args: None)
    aux.plot_final(str(tmp_path), str(path), None, [[1.0], [0.1]], 0, np.radians(30))
    fig = aux.pp.gcf()
    ax = fig.axes[0]
    red_legend = next(item for item in ax.artists if isinstance(item, Legend))
    handles = red_legend.get_lines()
    assert len(handles) == 3 and all(isinstance(handle, Line2D) for handle in handles)
    assert len({str(handle._unscaled_dash_pattern) for handle in handles}) == 3
    assert (tmp_path / "range_map_bns_m1_1.4_m2_1.4.pdf").stat().st_size > 1000
    aux.pp.close = monkeypatch.undo() or aux.pp.close


def test_targeted_range_legend_uses_selected_prior_and_solid_mass_handles():
    import targ_ac_git.aux_snr_mf as aux
    from targ_ac_git.targ_range_snr_mf import parse_iota_ranges
    aux.pp.figure()
    aux._add_tdr_legend([(1, 1), (1.4, 1.4), (2, 2)], parse_iota_ranges(10, 35))
    legend = aux.pp.gca().get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert len(labels) == 3 and all(label.startswith("m1=") for label in labels)
    assert legend.get_title().get_text() == r"$10^\circ\leq\iota\leq35^\circ$"
    assert all(isinstance(handle, aux.Line2D) for handle in legend.get_lines())
    assert not any(line.is_dashed() for line in legend.get_lines())
    aux.pp.close()


@pytest.mark.parametrize("bounds", [(10, 35), (0, 90)])
def test_targeted_pdfs_plot_three_solid_curves_and_json_keeps_all_priors(tmp_path, monkeypatch, bounds):
    import targ_ac_git.aux_snr_mf as aux
    from targ_ac_git.targ_range_snr_mf import CHIRP_MASSES, parse_iota_ranges

    (tmp_path / "results").mkdir()
    for family, masses in CHIRP_MASSES.items():
        for m1, m2 in masses:
            for tag in (["_SFHo", "_DD2"] if family == "nsbh" else [""]):
                filename = tmp_path / "results" / f"results_{family}{tag}_m1_{m1}_m2_{m2}.hdf"
                with h5py.File(filename, "w") as handle:
                    for field, value in {"optimal_snr_H1": 25., "ra": 1., "dec": .3, "tc": 1266908331.806}.items():
                        handle[field] = [value]

    figures = {}
    savefig = aux.pp.savefig

    def capture(filename, *args, **kwargs):
        figures[Path(filename).stem] = aux.pp.gcf()
        return savefig(filename, *args, **kwargs)

    monkeypatch.setattr(aux.pp, "savefig", capture)
    priors = parse_iota_ranges(*bounds)
    aux.compute_range([1.], [.3], ["H1"], CHIRP_MASSES, str(tmp_path), priors, 9, "optimal")
    for family in CHIRP_MASSES:
        ax = figures[f"{family}_targeted_range"].axes[0]
        assert len(ax.lines) == 3
        assert not any(line.is_dashed() for line in ax.lines)
        payload = json.loads((tmp_path / f"results_{family}.json").read_text())[family]
        assert len(payload) == 3
        assert all(set(entry["tdr"]) == {prior["label"] for prior in priors}
                   for entry in payload.values())


@pytest.mark.parametrize("offsets", [(0, 32), (32, 64)])
def test_evolution_legend_replaces_title_and_sits_above_axes(tmp_path, monkeypatch, offsets):
    import targ_ac_git.plot_time_evolution as plotting

    summary = {"grb_name": "GRBtest", "scan_step": 32, "skymap_file": "map.fit",
               "iota_min_deg": 0, "iota_max_deg": 45, "snr_threshold": 9}
    (tmp_path / "scan_summary.json").write_text(json.dumps(summary))
    rows = [{"offset_s": offset, "status": "complete", "bns": 100., "nsbh": 200.,
             "antenna_factor": {"network": .5}, "online_ifos": ["H1", "L1"]}
            for offset in offsets]
    monkeypatch.setattr(plotting, "load_rows", lambda *args: rows)
    closed = []
    close = plotting.plt.close
    monkeypatch.setattr(plotting.plt, "close", lambda fig: closed.append(fig))
    plotting.plot_time_evolution(str(tmp_path))
    fig = closed[-1]
    ax = fig.axes[0]
    fig.canvas.draw()
    assert ax.get_title() == ""
    assert ax.get_legend().get_window_extent().y0 > ax.get_window_extent().y1
    close(fig)
