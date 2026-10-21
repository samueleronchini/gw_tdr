import importlib
import json

import healpy as hp
import numpy as np
import pytest

from targ_ac_git import run_grb

runner = importlib.import_module("targ_ac_git.targ_range_snr_mf")


def _stub_analysis(monkeypatch):
    observed = []
    monkeypatch.setattr(runner, "_get_run_config", lambda t0: {"name": "test"})

    def single(center, output, ra, dec, skymap, cache, log, priors,
               map_min, map_max, threshold, snr_type):
        observed.append((center, priors, map_min, map_max, threshold, snr_type))
        return {"status": "complete", "online_ifos": ["H1"],
                "antenna_factor": {"network": 0.5}}

    monkeypatch.setattr(runner, "run_single_trigger", single)
    plotting = importlib.import_module("targ_ac_git.plot_time_evolution")
    monkeypatch.setattr(plotting, "plot_time_evolution", lambda output: None)
    return observed


def test_python_api_returns_summary_and_applies_defaults(tmp_path, monkeypatch):
    observed = _stub_analysis(monkeypatch)
    summary = run_grb(grb_name="GRB200228291", t0=1266908331.806,
                      output_dir=tmp_path, ra=333.89292, dec=-42.944167,
                      scan_start=-32, scan_end=3.584)
    assert [bin["offset_s"] for bin in summary["bins"]] == [-32, 0, 3.584]
    assert summary["user_prior"] == "0-45"
    assert summary["snr_threshold"] == 9
    assert "t90" not in summary
    assert summary == json.loads((tmp_path / "scan_summary.json").read_text())
    assert np.allclose([row[0] for row in observed],
                       1266908331.806 + np.array([-32, 0, 3.584]))
    assert [prior["label"] for prior in observed[0][1]] == ["0-45", "0-90"]
    assert np.isclose(observed[0][3], np.pi / 4)


def test_t90_is_no_longer_an_input(tmp_path):
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([
            "--output-dir", str(tmp_path), "--t0", "1266908331.806",
            "--ra", "0", "--dec", "0", "--t90", "3.584",
        ])
    with pytest.raises(TypeError, match="t90"):
        run_grb(grb_name="GRBtest", t0=1266908331.806,
                output_dir=tmp_path, ra=0, dec=0, t90=3.584)


def test_existing_command_line_returns_success_without_grb_name(tmp_path, monkeypatch):
    _stub_analysis(monkeypatch)
    monkeypatch.setattr("sys.argv", ["tdr", "--output-dir", str(tmp_path),
                                    "--t0", "1266908331.806", "--ra", "333.89292",
                                    "--dec=-42.944167"])
    # Console scripts pass this return value to sys.exit; it must not be a dict.
    assert runner.targ_range() is None
    summary = json.loads((tmp_path / "scan_summary.json").read_text())
    assert summary["grb_name"] == "GRB"
    assert len(summary["bins"]) == 1


def test_failed_bin_is_reported_and_raises(tmp_path, monkeypatch):
    _stub_analysis(monkeypatch)

    def fail(*args):
        raise RuntimeError("test strain failure")

    monkeypatch.setattr(runner, "run_single_trigger", fail)
    with pytest.raises(RuntimeError, match="Failed scan bins"):
        run_grb(grb_name="GRBtest", t0=1266908331.806, output_dir=tmp_path,
                ra=0, dec=0)
    summary = json.loads((tmp_path / "scan_summary.json").read_text())
    assert summary["bins"][0]["status"] == "failed"
    assert "test strain failure" in summary["bins"][0]["error"]


def test_color_antenna_is_area_mean_not_maximum_or_probability_mean(tmp_path):
    from pycbc.detector import Detector
    from targ_ac_git.aux_snr_mf import compute_antennamap
    from targ_ac_git.plot_time_evolution import compute_plot_antenna

    gps = 1266908331.806
    theta, ra = hp.pix2ang(8, np.arange(hp.nside2npix(8)), nest=True)
    dec = np.pi / 2 - theta
    per_ifo = {}
    for ifo in ("H1", "L1"):
        fp, fc = Detector(ifo).antenna_pattern(ra, dec, 0, gps)
        per_ifo[ifo] = np.hypot(fp, fc)
    network = np.sqrt(sum(values**2 for values in per_ifo.values()))
    order = np.argsort(network)
    pixels = [order[20], order[-20]]
    probability = np.zeros_like(network)
    probability[pixels] = [0.8, 0.15]
    probability[order[100]] = 0.05
    path = tmp_path / "map.fit"
    hp.write_map(path, probability, nest=True, dtype=float,
                 column_names=["PROBABILITY"])
    reference = compute_antennamap(["H1", "L1"], gps)
    actual = compute_plot_antenna(str(path), 0, 0, gps, ["H1", "L1"], reference)
    assert actual["region_pixel_count"] == 2
    assert np.isclose(actual["network"], network[pixels].mean())
    assert not np.isclose(actual["network"], network[pixels].max())
    assert not np.isclose(actual["network"],
                          np.average(network[pixels], weights=[0.8, 0.15]))
    assert not np.isclose(actual["network"], np.sqrt(np.mean(network[pixels]**2)))
    for ifo in per_ifo:
        assert np.isclose(actual["per_ifo"][ifo]["F_mean"], per_ifo[ifo][pixels].mean())
    assert actual["D90_sky_treatment"] == "probability_weighted_sky_samples"
