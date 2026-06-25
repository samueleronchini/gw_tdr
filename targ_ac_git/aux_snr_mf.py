import contextlib
import io
import json
import logging
import os
import subprocess

import h5py
import healpy as hp
import ligo.skymap.io
import ligo.skymap.plot
import ligo.skymap.postprocess
import matplotlib
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D
from pycbc.detector import Detector

matplotlib.use("Agg")
import matplotlib.pyplot as pp

pp.rcdefaults()
pp.style.use("default")


NSBH_LAMBDA2 = {
    "SFHo": {1.0: 3160, 1.4: 436, 2.0: 16},
    "DD2": {1.0: 4970, 1.4: 829, 2.0: 76},
}

NSBH_MIN_CHI1 = {
    "SFHo": {(5.0, 1.0): 0.0, (10.0, 1.4): 0.64638, (20.0, 2.0): 0.98617},
    "DD2": {(5.0, 1.0): 0.0, (10.0, 1.4): 0.51135, (20.0, 2.0): 0.93732},
}

NSIDE = 16
NPIX = hp.nside2npix(NSIDE)
THETA_PIX, PHI_PIX = hp.pix2ang(NSIDE, np.arange(NPIX), nest=True)
RA_PIX = np.degrees(PHI_PIX)
DEC_PIX = 90.0 - np.degrees(THETA_PIX)

MASS_COLORS = ["blue", "red", "green"]


def sample_isotropic_iota(iota_min, iota_max, size):
    """Sample inclination angles isotropically between two limits in radians."""
    if iota_min < 0:
        raise ValueError("iota_min must be >= 0")
    if iota_max > np.pi / 2:
        raise ValueError("iota_max must be <= pi/2 for the current convention")
    if iota_min >= iota_max:
        raise ValueError("iota_min must be smaller than iota_max")

    return np.arccos(np.random.uniform(np.cos(iota_max), np.cos(iota_min), size))


def build_injection_config(cbc_type, t0, position, m1, m2, iota, eos="SFHo"):
    """Return a PyCBC injection configuration string for a BNS or NSBH system."""
    if cbc_type == "bns":
        waveform = "TaylorT4"
        extra_params = """
spin1x = 0
spin1y = 0
spin1z = 0
spin2x = 0
spin2y = 0
spin2z = 0
lambda1 = 0
lambda2 = 0
"""

    elif cbc_type == "nsbh":
        waveform = "IMRPhenomNSBH"
        eos = str(eos)
        m1_key, m2_key = float(m1), float(m2)
        mass_pair = (m1_key, m2_key)

        if eos not in NSBH_LAMBDA2:
            raise ValueError(f"Unknown EOS '{eos}'. Choose from {list(NSBH_LAMBDA2.keys())}")
        if m2_key not in NSBH_LAMBDA2[eos]:
            raise ValueError(f"No lambda2 value for NS mass {m2_key} and EOS {eos}")
        if mass_pair not in NSBH_MIN_CHI1[eos]:
            raise ValueError(f"No minimum chi1 value for mass pair {mass_pair} and EOS {eos}")

        lambda2 = NSBH_LAMBDA2[eos][m2_key]
        chi1 = NSBH_MIN_CHI1[eos][mass_pair]

        extra_params = f"""
spin1x = 0
spin1y = 0
spin1z = {chi1}
spin2x = 0
spin2y = 0
spin2z = 0.0
lambda1 = 0
lambda2 = {lambda2}
"""
    else:
        raise ValueError(f"Unknown cbc_type: {cbc_type}")

    return f"""
[variable_params]

[static_params]
tc = {t0}
f_lower = 20
approximant = {waveform}
distance = 100
mass1 = {m1}
mass2 = {m2}
inclination = {iota}
polarization = 0
ra = {position[0]}
dec = {position[1]}
coa_phase = 0
{extra_params}
"""


def inject(cbc_type, t0, position, m1, m2, iota, output_dir, eos="SFHo"):
    """Create a one-event PyCBC injection file for the requested CBC system."""
    inj_dir = os.path.join(output_dir, "inj")
    os.makedirs(inj_dir, exist_ok=True)

    tag = f"_{eos}" if cbc_type == "nsbh" else ""
    config_file = os.path.join(inj_dir, f"inj_{cbc_type}{tag}_m1_{m1}_m2_{m2}.ini")
    output_file = os.path.join(inj_dir, f"injections_{cbc_type}{tag}_m1_{m1}_m2_{m2}.hdf")

    with open(config_file, "w") as f:
        f.write(build_injection_config(cbc_type, t0, position, m1, m2, iota, eos=eos))

    subprocess.run(["pycbc_create_injections", "--config-files", config_file, "--output-file", output_file, "--ninjections", "1", "--force"], check=True)


def plot_psd(psd_list, output_dir, ifos=None):
    """Plot the PSDs used for the available detector network."""
    os.makedirs(output_dir, exist_ok=True)
    if ifos is None:
        ifos = ["H1", "L1", "V1"][:len(psd_list)]

    pp.figure()
    for ifo, psd in zip(ifos, psd_list):
        if psd is not None:
            pp.loglog(psd.sample_frequencies, psd, label=f"{ifo} PSD")

    pp.grid()
    pp.legend()
    pp.title("Power Spectral Densities")
    pp.xlabel("Frequency (Hz)")
    pp.ylabel("PSD")
    pp.xlim(10, 5000)
    pp.ylim(1e-50, 1e-36)
    pp.tight_layout()
    pp.savefig(os.path.join(output_dir, "psd_plot.pdf"))
    pp.close()


def _compute_d90_from_snr_reference(final_snr_ref, distances, snr_threshold, required_fraction=0.9):
    """Compute D90 from a reference SNR distribution evaluated at 100 Mpc."""
    rescaled_snr_matrix = final_snr_ref[:, None] * (100.0 / distances[None, :])
    fraction_above = np.sum(rescaled_snr_matrix > snr_threshold, axis=0) / final_snr_ref.shape[0]
    below_threshold = fraction_above < required_fraction

    if np.any(below_threshold):
        return float(distances[np.where(below_threshold)[0][0]])
    return None


def _convert_opt_to_mf_snr(final_snr_opt, n_ifo, k_min=0.9, k_max=1.0):
    """
    Convert optimal network SNR to a matched-filter-like network SNR.

    The conversion samples:
        X ~ noncentral_chisquare(df=2*n_ifo, nonc=(k*rho_opt)^2)
        rho_mf = sqrt(X)
    with k uniformly distributed between k_min and k_max.
    """
    k = np.random.uniform(k_min, k_max, size=np.shape(final_snr_opt))
    nonc = (k * final_snr_opt) ** 2
    return np.sqrt(np.random.noncentral_chisquare(df=2 * n_ifo, nonc=nonc))


def _select_snr(final_snr_opt, final_snr_mf, snr_statistic):
    if snr_statistic == "matched_filter":
        return final_snr_mf
    if snr_statistic == "optimal":
        return final_snr_opt
    raise ValueError(f"Unknown snr_statistic: {snr_statistic}")


def _fraction_curve(final_snr_ref, distances, snr_threshold):
    detectable_distances = 100.0 * final_snr_ref / snr_threshold
    return np.array([np.mean(detectable_distances > dist) for dist in distances])


def _line_style_and_label(iota_ranges, prior_label, mass_label):
    if len(iota_ranges) == 1:
        return "-", mass_label
    if prior_label == iota_ranges[0]["label"]:
        return "--", None
    return "-", mass_label


def _add_tdr_legend(masses_list, iota_ranges):
    mass_handles = [
        Line2D([0], [0], color=MASS_COLORS[i % len(MASS_COLORS)], lw=2.0, label=f"m1={m1}, m2={m2}")
        for i, (m1, m2) in enumerate(masses_list)
    ]

    if len(iota_ranges) == 1:
        handles = mass_handles
    else:
        iota_handles = []
        for i, prior in enumerate(iota_ranges[:2]):
            iota_max = round(np.degrees(prior["iota_max"]))
            iota_handles.append(Line2D([0], [0], color="0.6", lw=2.0, linestyle="--" if i == 0 else "-", label=rf"$0 < \iota < {iota_max}^\circ$"))
        handles = mass_handles + iota_handles

    leg = pp.legend(handles=handles, fontsize=9, handlelength=3.0, borderpad=0.35, labelspacing=0.35, handletextpad=0.7)
    for line in leg.get_lines():
        line.set_linewidth(2.0)


def _read_optimal_snr_file(filename):
    ifos = []
    optimal_snr_data = {}

    with h5py.File(filename, "r") as f:
        for field in f.keys():
            if "optimal_snr" not in field:
                continue
            ifo = field.split("_")[-1]
            ifos.append(ifo)
            optimal_snr_data[ifo] = f[field][:]

        ra_inj = f["ra"][0]
        dec_inj = f["dec"][0]
        t0 = f["tc"][0]

    return ifos, optimal_snr_data, ra_inj, dec_inj, t0


def compute_range(ra, dec, online_ifos, chirp_masses, output_dir, iota_ranges, snr_threshold, snr_statistic, antenna_info=None):
    """
    Compute TDR values and range-fraction plots for BNS and NSBH systems.

    NSBH systems are computed for SFHo and DD2 internally, but the saved JSON and
    the plot report the EOS-averaged value/curve.
    """
    n_sample = 10000
    required_fraction = 0.9
    snr_threshold = float(snr_threshold)
    eos_list = ["SFHo", "DD2"]

    if snr_statistic not in ["matched_filter", "optimal"]:
        raise ValueError(f"Unknown snr_statistic='{snr_statistic}'. Use 'matched_filter' or 'optimal'.")

    snr_label = r"$\rho_{\rm MF}$" if snr_statistic == "matched_filter" else r"$\rho_{\rm opt}$"
    pol = np.random.uniform(0, 2 * np.pi, n_sample)
    iota_samples = {prior["label"]: sample_isotropic_iota(prior["iota_min"], prior["iota_max"], n_sample) for prior in iota_ranges}
    distances = np.logspace(0, np.log10(5000), 5000)
    detectors = {ifo: Detector(ifo) for ifo in online_ifos}

    for cbc_type, masses_list in chirp_masses.items():
        median_dist = []

        pp.figure()
        pp.xscale("log")
        pp.xlabel("Distance (Mpc)", fontsize=15)
        pp.ylabel(rf"Fraction of sources with {snr_label} $>$ {snr_threshold:g}", fontsize=15)
        pp.grid()

        if cbc_type == "nsbh":
            results_data = {"nsbh": {eos: {} for eos in eos_list}}
            eos_values = eos_list
        else:
            results_data = {cbc_type: {}}
            eos_values = [None]

        nsbh_plot_curves = {}

        for mass_index, (m1, m2) in enumerate(masses_list):
            mass_label = f"m1={m1}, m2={m2}"
            color = MASS_COLORS[mass_index % len(MASS_COLORS)]

            for eos in eos_values:
                tag = f"_{eos}" if eos is not None else ""
                filename = os.path.join(output_dir, f"results/results_{cbc_type}{tag}_m1_{m1}_m2_{m2}.hdf")

                if not os.path.exists(filename):
                    logging.info(f"Missing result file, skipping: {filename}")
                    continue

                ifos, optimal_snr_data, ra_inj, dec_inj, t0 = _read_optimal_snr_file(filename)
                n_ifo = len(ifos)

                if cbc_type == "nsbh":
                    results_data["nsbh"][eos][mass_label] = {
                        "online_ifos": sorted(ifos),
                        "eos": eos,
                        "lambda2": NSBH_LAMBDA2[eos][float(m2)],
                        "chi1_min": NSBH_MIN_CHI1[eos][(float(m1), float(m2))],
                        "snr_statistic": snr_statistic,
                        "snr_threshold": snr_threshold,
                        "tdr": {},
                    }
                    
                    if antenna_info is not None:
                        results_data["nsbh"][eos][mass_label].update(antenna_info)
                    
                    result_entry = results_data["nsbh"][eos][mass_label]
                else:
                    results_data[cbc_type][mass_label] = {
                        "online_ifos": sorted(ifos),
                        "snr_statistic": snr_statistic,
                        "snr_threshold": snr_threshold,
                        "tdr": {},
                    }
                    
                    if antenna_info is not None:
                        results_data[cbc_type][mass_label].update(antenna_info)
                    
                    result_entry = results_data[cbc_type][mass_label]

                for prior in iota_ranges:
                    prior_label = prior["label"]
                    iota = iota_samples[prior_label]

                    ra_use = np.array(ra) * np.ones(n_sample) if len(ra) != n_sample else np.array(ra)
                    dec_use = np.array(dec) * np.ones(n_sample) if len(dec) != n_sample else np.array(dec)
                    snr_distr = np.zeros((len(ifos), n_sample))

                    for ifo in ifos:
                        eff_dist = detectors[ifo].effective_distance(100, ra_use, dec_use, pol, t0, iota)
                        eff_dist_ref = detectors[ifo].effective_distance(100, ra_inj, dec_inj, 0, t0, 0)
                        snr_distr[ifos.index(ifo), :] = optimal_snr_data[ifo] * (eff_dist / eff_dist_ref) ** (-1)

                    final_snr_opt = np.sqrt(np.sum(np.square(snr_distr), axis=0))
                    final_snr_mf = _convert_opt_to_mf_snr(final_snr_opt, n_ifo)
                    final_snr_use = _select_snr(final_snr_opt, final_snr_mf, snr_statistic)

                    d90 = _compute_d90_from_snr_reference(final_snr_use, distances, snr_threshold, required_fraction)
                    result_entry["tdr"][prior_label] = {
                        "iota_min_deg": round(np.degrees(prior["iota_min"]), 2),
                        "iota_max_deg": round(np.degrees(prior["iota_max"]), 2),
                        "D90_Mpc": d90,
                    }

                    fraction_snr_gt_thr = _fraction_curve(final_snr_use, distances, snr_threshold)

                    if cbc_type == "nsbh":
                        nsbh_plot_curves.setdefault((m1, m2, prior_label), {})[eos] = fraction_snr_gt_thr
                        continue

                    below_threshold = fraction_snr_gt_thr < required_fraction
                    if np.any(below_threshold):
                        median_dist.append(distances[np.where(below_threshold)[0][0]])

                    line_style, label = _line_style_and_label(iota_ranges, prior_label, mass_label)
                    pp.plot(distances, fraction_snr_gt_thr, linestyle=line_style, color=color, linewidth=2.0, label=label)

        if cbc_type == "nsbh":
            avg_results = {"nsbh": {}}

            for mass_index, (m1, m2) in enumerate(masses_list):
                mass_label = f"m1={m1}, m2={m2}"
                color = MASS_COLORS[mass_index % len(MASS_COLORS)]
                avg_results["nsbh"][mass_label] = {
                    "online_ifos": [],
                    "snr_statistic": snr_statistic,
                    "snr_threshold": snr_threshold,
                    "eos_average": True,
                    "eos_used": eos_list,
                    "tdr": {},
                }
                
                if antenna_info is not None:
                    avg_results["nsbh"][mass_label].update(antenna_info)

                all_ifos = []

                for prior in iota_ranges:
                    prior_label = prior["label"]
                    d90_values = []

                    for eos in eos_list:
                        try:
                            eos_entry = results_data["nsbh"][eos][mass_label]
                            val = eos_entry["tdr"][prior_label]["D90_Mpc"]
                            if val is not None and np.isfinite(val):
                                d90_values.append(val)
                            all_ifos.extend(eos_entry.get("online_ifos", []))
                        except KeyError:
                            continue

                    avg_results["nsbh"][mass_label]["tdr"][prior_label] = {
                        "iota_min_deg": round(np.degrees(prior["iota_min"]), 2),
                        "iota_max_deg": round(np.degrees(prior["iota_max"]), 2),
                        "D90_Mpc": float(np.mean(d90_values)) if d90_values else None,
                    }

                    key = (m1, m2, prior_label)
                    if key not in nsbh_plot_curves:
                        logging.info(f"No NSBH plotting curves found for {mass_label}, prior={prior_label}")
                        continue

                    available_eos = [eos for eos in eos_list if eos in nsbh_plot_curves[key]]
                    if not available_eos:
                        continue

                    avg_curve = np.mean([nsbh_plot_curves[key][eos] for eos in available_eos], axis=0)
                    below_threshold = avg_curve < required_fraction
                    if np.any(below_threshold):
                        median_dist.append(distances[np.where(below_threshold)[0][0]])

                    line_style, label = _line_style_and_label(iota_ranges, prior_label, mass_label)
                    pp.plot(distances, avg_curve, linestyle=line_style, color=color, linewidth=2.0, label=label)

                avg_results["nsbh"][mass_label]["online_ifos"] = sorted(set(all_ifos))

            results_data = avg_results

        if median_dist:
            pp.xlim(0.1 * np.mean(median_dist), 10 * np.mean(median_dist))

        pp.grid(True, which="major", alpha=0.3)
        _add_tdr_legend(masses_list, iota_ranges)

        pp.tight_layout()
        pp.savefig(os.path.join(output_dir, f"{cbc_type}_targeted_range.pdf"))
        pp.close()

        json_output_path = os.path.join(output_dir, f"results_{cbc_type}.json")
        with open(json_output_path, "w") as f:
            json.dump(results_data, f, indent=4)

        logging.info(f"Results saved to {json_output_path}")

def compute_localization_antenna_factor(online_ifos, t0, ra_samples, dec_samples, mode):
    """
    Compute the network antenna factor for a GRB localization.

    If one sky position is supplied, this is the antenna factor at the source.
    If many samples are supplied, this returns the maximum antenna factor among
    the supplied localization samples.

    Definition:
        A = sqrt( sum_i (Fplus_i^2 + Fcross_i^2) / N_ifo )

    This normalization keeps the antenna factor on a roughly detector-network
    comparable scale.
    """
    if not online_ifos:
        return {
            "antenna_factor": None,
            "antenna_factor_mode": mode,
            "ra_deg": None,
            "dec_deg": None,
        }

    ra_array = np.asarray(ra_samples, dtype=float)
    dec_array = np.asarray(dec_samples, dtype=float)

    if ra_array.ndim == 0:
        ra_array = np.array([float(ra_array)])
    if dec_array.ndim == 0:
        dec_array = np.array([float(dec_array)])

    if len(ra_array) != len(dec_array):
        raise ValueError("ra_samples and dec_samples must have the same length")

    antenna_power = np.zeros(len(ra_array), dtype=float)

    for ifo in online_ifos:
        fp, fc = Detector(ifo).antenna_pattern(ra_array, dec_array, 0.0, t0)
        antenna_power += fp**2 + fc**2

    antenna_factor = np.sqrt(antenna_power / float(len(online_ifos)))

    best_index = int(np.nanargmax(antenna_factor))

    return {
        "antenna_factor": float(antenna_factor[best_index]),
        "antenna_factor_mode": mode,
        "antenna_factor_ra_deg": float(np.degrees(ra_array[best_index])),
        "antenna_factor_dec_deg": float(np.degrees(dec_array[best_index])),
        "antenna_factor_definition": "sqrt(sum_i(Fplus_i^2 + Fcross_i^2) / N_ifo)",
    }

def compute_antennamap(online_ifos, t0):
    ant_pat_map = np.zeros(len(RA_PIX))

    for ifo in online_ifos:
        fp, fc = Detector(ifo).antenna_pattern(np.radians(RA_PIX), np.radians(DEC_PIX), 0, t0)
        ant_pat_map += fp**2 + fc**2

    max_pixel_index = np.argmax(ant_pat_map)
    max_theta, max_phi = hp.pix2ang(NSIDE, max_pixel_index, nest=True)

    return np.radians(np.degrees(max_phi)), np.radians(90.0 - np.degrees(max_theta))


def compute_map(cbc_type, m1, m2, online_ifos, output_dir, iota_min, iota_max, snr_threshold, snr_statistic):
    n_sample = 1000
    required_fraction = 0.9
    snr_threshold = float(snr_threshold)
    chunk_size = 256

    iota_samples = sample_isotropic_iota(iota_min, iota_max, n_sample)
    pol = np.random.uniform(0, 2 * np.pi, n_sample)
    distances = np.logspace(0, np.log10(5000), 5000)
    filename = os.path.join(output_dir, f"results/results_{cbc_type}_m1_{m1}_m2_{m2}.hdf")

    detectors = {ifo: Detector(ifo) for ifo in online_ifos}
    ifos, optimal_snr_data, ra_inj, dec_inj, t0 = _read_optimal_snr_file(filename)

    eff_dist_ref = {ifo: detectors[ifo].effective_distance(100, ra_inj, dec_inj, 0, t0, 0) for ifo in ifos}
    map_data = np.zeros_like(RA_PIX, dtype=np.float64)
    k_idx = int(np.floor((1.0 - required_fraction) * n_sample))

    for start in range(0, len(RA_PIX), chunk_size):
        stop = min(start + chunk_size, len(RA_PIX))
        n_chunk = stop - start

        ra_array = np.radians(np.repeat(RA_PIX[start:stop], n_sample))
        dec_array = np.radians(np.repeat(DEC_PIX[start:stop], n_sample))
        pol_array = np.tile(pol, n_chunk)
        iota_array = np.tile(iota_samples, n_chunk)
        final_snr_sq = np.zeros((n_sample, n_chunk), dtype=np.float64)

        for ifo in ifos:
            eff_dist = detectors[ifo].effective_distance(100, ra_array, dec_array, pol_array, t0, iota_array).reshape(n_chunk, n_sample).T
            final_snr_sq += (optimal_snr_data[ifo][:, None] * (eff_dist / eff_dist_ref[ifo]) ** (-1)) ** 2

        final_snr_opt = np.sqrt(final_snr_sq)
        final_snr_mf = _convert_opt_to_mf_snr(final_snr_opt, len(ifos))
        final_snr_use = _select_snr(final_snr_opt, final_snr_mf, snr_statistic)

        detectable_distance = 100.0 * final_snr_use / snr_threshold
        critical_distance = np.partition(detectable_distance, k_idx, axis=0)[k_idx, :]
        idx = np.searchsorted(distances, critical_distance, side="left")

        chunk_map = np.zeros(n_chunk, dtype=np.float64)
        valid = idx < len(distances)
        chunk_map[valid] = distances[idx[valid]]
        map_data[start:stop] = chunk_map

    outpath = os.path.join(output_dir, f"range_map_{cbc_type}_m1_{m1}_m2_{m2}.fits")
    with contextlib.redirect_stdout(io.StringIO()):
        hp.write_map(outpath, map_data, nest=True, overwrite=True)

    return outpath


def check_skymap(skymap):
    """Return a skymap object readable by ligo.skymap, renaming PROBABILITY to PROB if needed."""
    with fits.open(skymap) as hdulist:
        for hdu in hdulist:
            data = getattr(hdu, "data", None)
            cols = getattr(hdu, "columns", None)

            if data is None or cols is None:
                continue

            colnames = list(cols.names) if cols.names is not None else []
            if "PROB" in colnames:
                return skymap

            if "PROBABILITY" in colnames:
                new_cols = []
                for col in cols:
                    if col.name == "PROBABILITY":
                        col = fits.Column(name="PROB", format=col.format, unit=col.unit, array=data["PROBABILITY"])
                    new_cols.append(col)
                return fits.BinTableHDU.from_columns(new_cols)

    raise ValueError(f"{skymap} is not a table-based HEALPix skymap with PROB/PROBABILITY columns")


def map_samples(skymap):
    try:
        prob, _ = ligo.skymap.io.fits.read_sky_map(skymap, nest=True, distances=False)
    except Exception as e:
        logging.info(f"Error reading skymap: {e}; trying to rename PROBABILITY to PROB")
        prob, _ = ligo.skymap.io.fits.read_sky_map(check_skymap(skymap), nest=True, distances=False)

    prob = np.asarray(prob, dtype=float)
    prob[~np.isfinite(prob)] = 0.0
    prob = np.clip(prob, 0.0, None)

    if prob.sum() <= 0:
        raise ValueError(f"Skymap {skymap} has no positive probability after sanitization")

    prob /= prob.sum()
    nside = hp.npix2nside(len(prob))
    sampled_pixels = np.random.choice(np.arange(len(prob)), size=10000, p=prob)
    theta, phi = hp.pix2ang(nside, sampled_pixels, nest=True)

    return np.radians(np.degrees(phi)), np.radians(90.0 - np.degrees(theta))


def plot_final(output_dir, range_map, skymap, samples, iota_min, iota_max):
    fig = pp.figure()
    ax = pp.axes(projection="astro degrees mollweide")
    ax.grid()

    with fits.open(range_map) as hdulist:
        data = hdulist[1].data
        if "T" not in data.names:
            raise KeyError("Column 'T' not found in the FITS file.")
        hpx = data["T"].flatten()

    if skymap is not None:
        try:
            skymap_data, _ = ligo.skymap.io.fits.read_sky_map(skymap, nest=True, distances=False)
        except Exception as e:
            logging.info(f"Error reading skymap: {e}; trying to rename PROBABILITY to PROB")
            skymap_data, _ = ligo.skymap.io.fits.read_sky_map(check_skymap(skymap), nest=True, distances=False)

        cls = 100.0 * ligo.skymap.postprocess.util.find_greedy_credible_levels(skymap_data)
        ax.contour_hpx((cls, "ICRS"), nested=True, colors="black", levels=(50, 90), zorder=3, linestyles=["dashed", "solid"])

    ra_samples = np.degrees(samples[0])
    dec_samples = np.degrees(samples[1])

    if len(samples[0]) == 1:
        ax.scatter(ra_samples, dec_samples, marker="x", color="black", s=100, transform=ax.get_transform("icrs"), linewidths=1.5, zorder=2)
        black_handles = [Line2D([0], [0], color="black", marker="x", linestyle="None", markersize=9, markeredgewidth=1.5, label="GRB position")]
    else:
        ax.scatter(ra_samples, dec_samples, marker="x", color="gray", s=0.2, transform=ax.get_transform("icrs"), linewidths=0.2, alpha=1, zorder=1)
        black_handles = [
            Line2D([0], [0], color="black", linestyle="dashed", linewidth=2.2, label="GRB 50%"),
            Line2D([0], [0], color="black", linestyle="solid", linewidth=2.2, label="GRB 90%"),
        ]

    vmin, vmax = np.nanmin(hpx), np.nanmax(hpx)
    levels = [round(vmin + (vmax - vmin) / 4, -1), round(vmin + (vmax - vmin) / 2, -1), round(vmin + 3 * (vmax - vmin) / 4, -1)]

    ax.contour_hpx((hpx, "ICRS"), nested=True, colors="red", levels=levels, zorder=1, linestyles=["dotted", "dashdot", "solid"])
    ax.imshow_hpx((hpx, "ICRS"), cmap="GnBu_r", alpha=1.0, nested=True, zorder=0)

    red_handles = [
        Line2D([0], [0], color="red", linestyle="dotted", linewidth=2, label=f"{int(levels[0])} Mpc"),
        Line2D([0], [0], color="red", linestyle="dashdot", linewidth=2, label=f"{int(levels[1])} Mpc"),
        Line2D([0], [0], color="red", linestyle="solid", linewidth=2, label=f"{int(levels[2])} Mpc"),
    ]

    red_legend = ax.legend(handles=red_handles, loc="lower left", frameon=True, bbox_to_anchor=(-0.05, -0.22), borderaxespad=0.5, fontsize=9, handlelength=4.0, borderpad=0.4, labelspacing=0.4, handletextpad=0.8)
    ax.add_artist(red_legend)
    ax.legend(handles=black_handles, loc="lower right", frameon=True, bbox_to_anchor=(1.05, -0.2), borderaxespad=0.5, fontsize=9, handlelength=4.5, borderpad=0.4, labelspacing=0.4, handletextpad=0.8)

    sm = pp.cm.ScalarMappable(cmap=pp.cm.GnBu_r, norm=pp.Normalize(vmin=vmin, vmax=vmax))
    cbar = pp.colorbar(sm, ax=ax, shrink=1.0, orientation="horizontal", aspect=30)
    cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
    cbar.set_label("Targeted detectability range (Mpc)")

    parts = os.path.basename(range_map).replace(".fits", "").split("_")
    cbc_type, m1, m2 = parts[2], parts[4], parts[6]
    title = f"{cbc_type.upper()} ({m1} + {m2}) $M_{{\\odot}}$"
    title += rf", ${round(np.degrees(iota_min))}^\circ < \iota < {round(np.degrees(iota_max))}^\circ$"
    ax.set_title(title, fontsize=15)

    pp.tight_layout()
    pp.savefig(os.path.join(output_dir, f"range_map_{cbc_type}_m1_{m1}_m2_{m2}.pdf"))
    pp.close(fig)