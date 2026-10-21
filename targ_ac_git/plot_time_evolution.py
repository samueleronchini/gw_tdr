"""Automatic version of the supplied BNS/NSBH + IFO evolution plot."""
import argparse
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D

from .aux_snr_mf import DASHED, check_skymap


def _factor_at(ra, dec, t0, ifos):
    from pycbc.detector import Detector
    per_ifo = {}
    power = 0.0
    for ifo in sorted(ifos):
        fp, fc = Detector(ifo).antenna_pattern(ra, dec, 0, t0)
        fp, fc = float(fp), float(fc)
        per_ifo[ifo] = {"Fp": fp, "Fc": fc, "F": float(np.hypot(fp, fc))}
        power += fp**2 + fc**2
    return {"network": float(np.sqrt(power)), "per_ifo": per_ifo,
            "ra_deg": float(np.degrees(ra)), "dec_deg": float(np.degrees(dec))}


def compute_plot_antenna(skymap, ra, dec, t0, ifos, injection_reference):
    """Plot metadata only: never feeds a position or factor back into D90.

    The localization error region is the map's 90% highest-probability region,
    including the crossing pixel and boundary ties, at native pixel centers.
    """
    reference = _factor_at(*injection_reference, t0, ifos)
    if skymap is None:
        position = (ra, dec) if ra is not None and dec is not None else injection_reference
        result = _factor_at(*position, t0, ifos)
        result.update(position_basis="source_position", D90_sky_treatment="fixed_position")
    else:
        import healpy as hp
        import ligo.skymap.io.fits
        from pycbc.detector import Detector
        try:
            prob, _ = ligo.skymap.io.fits.read_sky_map(skymap, nest=True, distances=False)
        except Exception:
            prob, _ = ligo.skymap.io.fits.read_sky_map(check_skymap(skymap), nest=True, distances=False)
        prob = np.asarray(prob, dtype=float)
        prob[~np.isfinite(prob)] = 0
        prob = np.clip(prob, 0, None)
        if prob.sum() <= 0:
            raise ValueError("Localization map has no positive probability")
        prob /= prob.sum()
        order = np.argsort(-prob, kind="stable")
        crossing = min(np.searchsorted(np.cumsum(prob[order]), 0.9), len(order) - 1)
        pixels = np.flatnonzero((prob >= prob[order[crossing]]) & (prob > 0))
        nside = hp.npix2nside(len(prob))
        detectors = {ifo: Detector(ifo) for ifo in sorted(ifos)}
        network_sum = 0.0
        per_ifo_sum = {ifo: 0.0 for ifo in detectors}
        for start in range(0, len(pixels), 65536):
            current = pixels[start:start + 65536]
            theta, phi = hp.pix2ang(nside, current, nest=True)
            power = np.zeros(len(current))
            for ifo, detector in detectors.items():
                fp, fc = detector.antenna_pattern(phi, np.pi / 2 - theta, 0, t0)
                ifo_power = fp**2 + fc**2
                power += ifo_power
                per_ifo_sum[ifo] += float(np.sqrt(ifo_power).sum())
            network_sum += float(np.sqrt(power).sum())
        result = {
            "network": network_sum / len(pixels),
            "per_ifo": {
                ifo: {"F_mean": total / len(pixels)}
                for ifo, total in per_ifo_sum.items()
            },
            "position_basis": "area_mean_in_90_percent_localization_region",
            "averaging": "arithmetic_mean_over_equal_area_pixels",
            "credible_level": 0.9,
            "probability_enclosed": float(prob[pixels].sum()),
            "map_nside": int(nside),
            "region_pixel_count": int(len(pixels)),
            "D90_sky_treatment": "probability_weighted_sky_samples",
        }
    result["injection_reference"] = reference
    result["use"] = "JSON and evolution-plot metadata only; D90 calculation is unchanged"
    return result


def _mass_entry(data, m1, m2):
    for key, entry in data.items():
        match = re.fullmatch(r"m1=([\d.]+),\s*m2=([\d.]+)", key)
        if match and tuple(map(float, match.groups())) == (m1, m2):
            return entry
    return None


def load_rows(output_dir, summary):
    rows = []
    for item in summary["bins"]:
        row = dict(item, bns=np.nan, nsbh=np.nan)
        if item["status"] == "complete":
            for family, masses in (("bns", (1.4, 1.4)), ("nsbh", (10.0, 1.4))):
                path = os.path.join(output_dir, item["scan_name"], f"results_{family}.json")
                with open(path) as handle:
                    entry = _mass_entry(json.load(handle)[family], *masses)
                if entry is not None:
                    value = entry["tdr"][summary["user_prior"]]["D90_Mpc"]
                    row[family] = np.nan if value is None else float(value)
        rows.append(row)
    return sorted(rows, key=lambda row: row["offset_s"])


def plot_time_evolution(output_dir):
    with open(os.path.join(output_dir, "scan_summary.json")) as handle:
        summary = json.load(handle)
    rows = load_rows(output_dir, summary)
    offsets = np.asarray([row["offset_s"] for row in rows], dtype=float)
    if len(offsets) == 1:
        edges = [offsets[0] - summary["scan_step"] / 2, offsets[0] + summary["scan_step"] / 2]
    else:
        edges = [offsets[0], *list((offsets[:-1] + offsets[1:]) / 2), offsets[-1]]
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})
    fig = plt.figure(figsize=(8.2, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.045], height_ratios=[1.7, 0.75],
                          hspace=0.0, wspace=0.04)
    ax_d90 = fig.add_subplot(gs[0, 0])
    ax_det = fig.add_subplot(gs[1, 0], sharex=ax_d90)
    cax = fig.add_subplot(gs[0, 1])
    plt.setp(ax_d90.get_xticklabels(), visible=False)
    factors = [row["antenna_factor"]["network"] for row in rows if row["status"] == "complete"]
    vmin, vmax = (min(factors), max(factors)) if factors else (0.0, 1.0)
    if np.isclose(vmin, vmax):
        vmin, vmax = max(0.0, vmin - 0.05), vmax + 0.05
    norm = plt.Normalize(vmin, vmax)
    for family, marker in (("bns", "o"), ("nsbh", "s")):
        for row in rows:
            if np.isfinite(row[family]):
                ax_d90.scatter(row["offset_s"], row[family], marker=marker, s=70,
                                c=[row["antenna_factor"]["network"]], cmap="viridis", norm=norm,
                                edgecolors="black", linewidth=1.0, zorder=2)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax)
    bar.set_label(r"$\langle F_{\rm net}\rangle$ in 90% localization" if summary["skymap_file"]
                   else "Network antenna factor at source")
    all_valid = [(row[family], row["offset_s"]) for row in rows for family in ("bns", "nsbh")
                 if np.isfinite(row[family])]
    reference = 0 if any(off == 0 for _, off in all_valid) else (max(all_valid)[1] if all_valid else 0)
    for ax in (ax_d90, ax_det):
        ax.axvline(0, color="black", linestyle=DASHED, linewidth=1.2, alpha=0.9, zorder=0)
        if reference != 0:
            ax.axvline(reference, color="gray", linestyle=DASHED, linewidth=1.5, alpha=0.8, zorder=0)
    colors = {"H1": "#4C9FEF", "L1": "#F03B3B", "V1": "#30AD63"}
    for index, row in enumerate(rows):
        for detector, y in (("H1", 2), ("L1", 1), ("V1", 0)):
            if detector in (row.get("online_ifos") or []):
                if len(rows) == 1:
                    ax_det.scatter(row["offset_s"], y, marker="s", s=70, color=colors[detector])
                else:
                    ax_det.broken_barh([(edges[index], edges[index + 1] - edges[index])],
                                       (y - 0.275, 0.55), facecolors=colors[detector], edgecolors="none", alpha=0.95)
    ax_det.set_ylim(-0.7, 2.7)
    ax_det.set_yticks([0, 1, 2], labels=["V1", "L1", "H1"])
    ax_det.set_ylabel("IFO")
    ax_det.set_xlabel(r"Offset from $T_0$ (s)")
    ax_d90.set_ylabel(r"$D_{90}$ (Mpc)")
    ax_d90.grid(False)
    ax_det.grid(False)
    if all_valid:
        ax_d90.set_ylim(0, max(value for value, _ in all_valid) * 1.12)
    else:
        ax_d90.text(0.5, 0.5, "No valid D90 results", transform=ax_d90.transAxes, ha="center")
    pad = 0.02 * (edges[-1] - edges[0]) if len(rows) > 1 else 0
    ax_det.set_xlim(edges[0] - pad, edges[-1] + pad)
    handles = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=8, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.5, label=r"$\mathrm{NSBH}\ (10,1.4)\,M_\odot$"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.5, label=r"$\mathrm{BNS}\ (1.4,1.4)\,M_\odot$"),
        Line2D([0], [0], color="black", linestyle=DASHED, linewidth=1.2, label=r"$T_0$"),
    ]
    if reference != 0:
        handles.append(Line2D([0], [0], color="gray", linestyle=DASHED, linewidth=1.5,
                               label=f"Max D90: T={reference:g} s"))
    ax_d90.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                   ncol=len(handles), frameon=True, fontsize=10,
                   handler_map={Line2D: HandlerLine2D(numpoints=1)}, numpoints=1)
    prior = rf"${summary['iota_min_deg']:g}^\circ\leq\iota\leq{summary['iota_max_deg']:g}^\circ$"
    fig.text(0.12, 0.01, prior + f"; SNR threshold = {summary['snr_threshold']:g}", fontsize=9)
    fig.subplots_adjust(bottom=0.13, right=0.87, top=0.92)
    path = os.path.join(output_dir, f"{summary['grb_name']}_D90_and_IFO_windows_combined.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    print(plot_time_evolution(parser.parse_args().output_dir))
