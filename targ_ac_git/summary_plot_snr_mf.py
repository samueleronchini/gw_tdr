import json
import math
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.time import Time


DEFAULT_BNS_MASS_KEY = "m1=1.4, m2=1.4"
DEFAULT_NSBH_MASS_KEY = "m1=10, m2=1.4"


DETECTOR_Y = {
    "H1": 2,
    "L1": 1,
    "V1": 0,
}

DETECTOR_COLORS = {
    "H1": "#4C9FEF",
    "L1": "#F03B3B",
    "V1": "#9B59B6",
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def scan_dir_to_offset(scan_name):
    """
    Convert scan directory names to offsets in seconds.

    Examples
    --------
    scan_t0p0000 -> 0
    scan_t0m0020 -> -20
    scan_t0p0032 -> +32
    scan_t0p0012p500 -> +12.5
    """
    base = os.path.basename(scan_name.rstrip("/"))

    match = re.match(r"scan_t0([pm])(.+)$", base)
    if match is None:
        raise ValueError(f"Could not parse scan directory name: {scan_name}")

    sign = 1.0 if match.group(1) == "p" else -1.0
    value_text = match.group(2).replace("p", ".")

    return sign * float(value_text)


def load_timebin_summary(output_dir):
    """
    Load the multi-timebin metadata written by targ_range_snr_mf.py.

    This is preferred because it records the exact offsets generated from
    --t-start, --t-end, and --t-bin.
    """
    summary_path = os.path.join(output_dir, "multi_timebin_summary.json")

    if not os.path.exists(summary_path):
        return None

    return load_json(summary_path)


def discover_scan_dirs(output_dir):
    """
    Fallback if multi_timebin_summary.json is unavailable.
    """
    scan_dirs = []

    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and name.startswith("scan_t0"):
            scan_dirs.append(path)

    scan_dirs = sorted(scan_dirs, key=lambda p: scan_dir_to_offset(os.path.basename(p)))
    return scan_dirs


def build_scan_records(output_dir):
    """
    Return a list of dictionaries with:
      offset_s, t_center_gps, scan_name, scan_dir

    The primary source is multi_timebin_summary.json. If that file is absent,
    scan_t0... directories are discovered directly.
    """
    summary = load_timebin_summary(output_dir)

    if summary is not None:
        t0 = float(summary["t0"])
        records = []

        for item in summary.get("results", []):
            offset = float(item["offset"])
            scan_name = os.path.basename(item["output_dir"].rstrip("/"))
            scan_dir = os.path.join(output_dir, scan_name)

            records.append({
                "offset_s": offset,
                "t_center_gps": float(item.get("t_center", t0 + offset)),
                "scan_name": scan_name,
                "scan_dir": scan_dir,
            })

        records.sort(key=lambda r: r["offset_s"])
        return records, summary

    scan_dirs = discover_scan_dirs(output_dir)

    if len(scan_dirs) == 0:
        raise RuntimeError(f"No scan_t0... directories found in {output_dir}")

    records = []

    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)
        offset = scan_dir_to_offset(scan_name)

        records.append({
            "offset_s": offset,
            "t_center_gps": None,
            "scan_name": scan_name,
            "scan_dir": scan_dir,
        })

    return records, None


def normalize_ifo_label(value):
    if value is None:
        return "UNKNOWN"

    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return "".join(sorted(parts)) if parts else "UNKNOWN"

    text = str(value).strip()

    if text == "" or text.lower() == "none":
        return "UNKNOWN"

    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return "".join(sorted(parts)) if parts else "UNKNOWN"

    return text


def detectors_from_ifo_label(ifo_label):
    if ifo_label is None:
        return []

    text = str(ifo_label).strip()

    if text == "" or text.upper() == "UNKNOWN" or text.lower() == "none":
        return []

    detectors = []

    for det in ["H1", "L1", "V1"]:
        if det in text:
            detectors.append(det)

    return detectors


def get_antenna_factor(entry):
    """
    Support both the new GitHub-ready format:

        "antenna_factor": 0.52

    and older nested formats such as:

        "antenna_factor": {"network": 0.52}
    """
    val = entry.get("antenna_factor", np.nan)

    if isinstance(val, dict):
        for key in ["network", "source", "value"]:
            try:
                out = float(val[key])
                if np.isfinite(out):
                    return out
            except Exception:
                pass
        return np.nan

    try:
        out = float(val)
        return out if np.isfinite(out) else np.nan
    except Exception:
        return np.nan


def choose_iota_label(tdr_block, requested_iota_label=None):
    """
    Choose which inclination prior to plot.

    The main pipeline should pass requested_iota_label explicitly, based on
    the user's parsed --iota-min/--iota-max values.
    """
    if not isinstance(tdr_block, dict) or len(tdr_block) == 0:
        return None

    if requested_iota_label is not None:
        if requested_iota_label not in tdr_block:
            available = list(tdr_block.keys())
            raise KeyError(
                f"Requested iota label '{requested_iota_label}' not found. "
                f"Available labels: {available}"
            )

        return requested_iota_label

    if len(tdr_block) == 1:
        return list(tdr_block.keys())[0]

    available = list(tdr_block.keys())
    raise ValueError(
        "Multiple inclination priors found in the JSON, but no iota_label "
        f"was passed to the summary plot. Available labels: {available}"
    )

def describe_iota_label(rows):
    """
    Return a human-readable inclination-prior label from the plotted rows.
    """
    labels = sorted({
        str(r.get("iota_label"))
        for r in rows
        if r.get("iota_label") is not None
    })

    if len(labels) == 0:
        return "unknown inclination prior"

    if len(labels) == 1:
        label = labels[0]

        if "-" in label:
            try:
                iota_min, iota_max = label.split("-", 1)
                return rf"${float(iota_min):g}^\circ \leq \iota \leq {float(iota_max):g}^\circ$"
            except Exception:
                return label

        return label

    return ", ".join(labels)

def extract_family_result(json_path, family, mass_key, requested_iota_label=None):
    """
    Extract D90, IFOs, antenna factor, and iota label from one results_*.json file.
    """
    if not os.path.exists(json_path):
        return None

    data = load_json(json_path)

    try:
        mass_entry = data[family][mass_key]
    except KeyError:
        available = data.get(family, {}).keys()
        raise KeyError(
            f"Mass key '{mass_key}' not found in {json_path}. "
            f"Available mass keys: {list(available)}"
        )

    tdr_block = mass_entry.get("tdr", {})
    iota_label = choose_iota_label(tdr_block, requested_iota_label)

    if iota_label is None:
        return None

    d90 = tdr_block[iota_label].get("D90_Mpc", None)

    if d90 is None:
        d90 = np.nan
    else:
        d90 = float(d90)

    return {
        "d90": d90,
        "ifos": normalize_ifo_label(mass_entry.get("online_ifos")),
        "antenna_factor": get_antenna_factor(mass_entry),
        "iota_label": iota_label,
    }


def read_timebin_rows(
    output_dir,
    bns_mass_key=DEFAULT_BNS_MASS_KEY,
    nsbh_mass_key=DEFAULT_NSBH_MASS_KEY,
    iota_label=None,
):
    records, summary = build_scan_records(output_dir)

    bns_rows = []
    nsbh_rows = []

    for record in records:
        scan_dir = record["scan_dir"]

        bns_path = os.path.join(scan_dir, "results_bns.json")
        nsbh_path = os.path.join(scan_dir, "results_nsbh.json")

        bns_result = extract_family_result(
            json_path=bns_path,
            family="bns",
            mass_key=bns_mass_key,
            requested_iota_label=iota_label,
        )

        nsbh_result = extract_family_result(
            json_path=nsbh_path,
            family="nsbh",
            mass_key=nsbh_mass_key,
            requested_iota_label=iota_label,
        )

        if bns_result is not None:
            row = dict(record)
            row.update(bns_result)
            row["family"] = "bns"
            bns_rows.append(row)

        if nsbh_result is not None:
            row = dict(record)
            row.update(nsbh_result)
            row["family"] = "nsbh"
            nsbh_rows.append(row)

    bns_rows.sort(key=lambda r: r["offset_s"])
    nsbh_rows.sort(key=lambda r: r["offset_s"])

    return bns_rows, nsbh_rows, summary


def build_detector_intervals(rows, segment_half_width=5.0):
    """
    Build detector-availability intervals from the actual strain segment used
    for each timebin.

    Each timebin analyzes a 256 s strain segment:
        [offset_s - 5 s, offset_s + 5 s]

    If an IFO is listed as online/usable in that timebin, it is shown over
    that 256 s interval.
    """
    detector_intervals = {
        "H1": [],
        "L1": [],
        "V1": [],
    }

    for row in rows:
        offset = float(row["offset_s"])
        start = offset - segment_half_width
        width = 2.0 * segment_half_width

        for det in detectors_from_ifo_label(row["ifos"]):
            detector_intervals[det].append((start, width))

    return detector_intervals


def infer_window_limits(rows, summary, margin_s=5.0):
    """
    Set x-axis limits for the summary plot.

    The D90 points remain located at the actual timebin centers:
        -T_start, ..., 0, ..., +T_end

    The displayed x-axis range is extended slightly for visual padding:
        xmin = -T_start - margin_s
        xmax = +T_end   + margin_s
    """
    if summary is not None:
        t_start = float(summary.get("t_start", 0.0))
        t_end = float(summary.get("t_end", 0.0))
        return -t_start - margin_s, t_end + margin_s

    offsets = [float(r["offset_s"]) for r in rows]

    if len(offsets) == 0:
        return -margin_s, margin_s

    return min(offsets) - margin_s, max(offsets) + margin_s

def build_offset_xticks(rows, summary, window_min, window_max):
    """
    Build x ticks at multiples of T_bin only.

    Preferred source of T_bin is multi_timebin_summary.json.
    If unavailable, infer it from the spacing between offsets.
    """
    if summary is not None and "t_bin" in summary:
        t_bin = float(summary["t_bin"])
    else:
        offsets = sorted({float(r["offset_s"]) for r in rows})
        diffs = [
            abs(offsets[i + 1] - offsets[i])
            for i in range(len(offsets) - 1)
            if abs(offsets[i + 1] - offsets[i]) > 1e-9
        ]

        if len(diffs) == 0:
            t_bin = 1.0
        else:
            t_bin = min(diffs)

    if t_bin <= 0:
        t_bin = 1.0

    k_min = int(math.ceil(window_min / t_bin))
    k_max = int(math.floor(window_max / t_bin))

    tick_positions = [k * t_bin for k in range(k_min, k_max + 1)]

    tick_labels = []
    for x in tick_positions:
        if np.isclose(x, round(x)):
            tick_labels.append(f"{int(round(x))}")
        else:
            tick_labels.append(f"{x:g}")

    return tick_positions, tick_labels
    






def choose_xtick_rows(rows, max_ticks=9):
    """
    Avoid unreadable UTC tick labels when there are many timebins.
    Always include first, T0 if present, and last.
    """
    if len(rows) <= max_ticks:
        return rows

    offsets = np.array([float(r["offset_s"]) for r in rows])
    keep_indices = set(np.linspace(0, len(rows) - 1, max_ticks, dtype=int).tolist())

    t0_indices = np.where(np.isclose(offsets, 0.0))[0]
    if len(t0_indices) > 0:
        keep_indices.add(int(t0_indices[0]))

    keep_indices.add(0)
    keep_indices.add(len(rows) - 1)

    return [rows[i] for i in sorted(keep_indices)]


def finite_values(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def plot_family(ax, rows, family, vmin, vmax):
    if family == "bns":
        marker = "o"
    elif family == "nsbh":
        marker = "s"
    else:
        marker = "o"

    sc = None

    for row in rows:
        d90 = row["d90"]

        if not np.isfinite(d90):
            continue

        ant = row["antenna_factor"]

        sc = ax.scatter(
            row["offset_s"],
            d90,
            marker=marker,
            s=75,
            c=[ant],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            linewidth=1.0,
            zorder=3,
        )

    return sc


def plot_timebin_summary(
    output_dir,
    bns_mass_key=DEFAULT_BNS_MASS_KEY,
    nsbh_mass_key=DEFAULT_NSBH_MASS_KEY,
    iota_label=None,
    output_pdf=None,
):
    """
    Create a PDF summary plot for one GRB analyzed at multiple timebins.

    The plot contains:
      1. D90 versus timebin, for BNS and NSBH.
      2. Detector availability inferred from the online IFOs in each timebin.
      3. UTC labels on the x-axis, with offsets relative to T0.
    """
    bns_rows, nsbh_rows, summary = read_timebin_rows(
        output_dir=output_dir,
        bns_mass_key=bns_mass_key,
        nsbh_mass_key=nsbh_mass_key,
        iota_label=iota_label,
    )

    if len(bns_rows) == 0 and len(nsbh_rows) == 0:
        raise RuntimeError(f"No valid BNS/NSBH rows found in {output_dir}")

    all_rows = sorted(bns_rows + nsbh_rows, key=lambda r: r["offset_s"])
    iota_text = describe_iota_label(all_rows)

    # Use BNS rows for detector intervals when available because BNS/NSBH
    # should have the same online IFOs in a given timebin.
    detector_source_rows = bns_rows if len(bns_rows) > 0 else nsbh_rows

    window_min, window_max = infer_window_limits(
        detector_source_rows,
        summary,
        margin_s=5.0,
    )

    detector_intervals = build_detector_intervals(
        detector_source_rows, segment_half_width=128.0,
    )

    antenna_values = finite_values([r["antenna_factor"] for r in all_rows])

    if len(antenna_values) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(antenna_values))
        vmax = float(np.nanmax(antenna_values))

        if np.isclose(vmin, vmax):
            vmin -= 0.05
            vmax += 0.05

    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 10,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })

    fig = plt.figure(figsize=(8.8, 6.4))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 0.045],
        height_ratios=[1.7, 0.75],
        hspace=0.0,
        wspace=0.05,
    )

    ax_d90 = fig.add_subplot(gs[0, 0])
    ax_det = fig.add_subplot(gs[1, 0], sharex=ax_d90)
    cax = fig.add_subplot(gs[0, 1])

    plt.setp(ax_d90.get_xticklabels(), visible=False)

    for ax in [ax_d90, ax_det]:
        ax.axvline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            zorder=1,
        )

    sc_bns = plot_family(
        ax=ax_d90,
        rows=bns_rows,
        family="bns",
        vmin=vmin,
        vmax=vmax,
    )

    sc_nsbh = plot_family(
        ax=ax_d90,
        rows=nsbh_rows,
        family="nsbh",
        vmin=vmin,
        vmax=vmax,
    )

    sc = sc_nsbh if sc_nsbh is not None else sc_bns

    ax_d90.set_ylabel(r"$D_{90}$ (Mpc)")
    ax_d90.grid(axis="y", alpha=0.25)

    if sc is not None:
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("Network antenna factor")

    bar_height = 0.55

    for det in ["H1", "L1", "V1"]:
        intervals = detector_intervals[det]

        if len(intervals) == 0:
            continue

        ax_det.broken_barh(
            intervals,
            (DETECTOR_Y[det] - bar_height / 2, bar_height),
            facecolors=DETECTOR_COLORS[det],
            edgecolors="none",
            alpha=0.95,
            zorder=2,
        )

    ax_det.set_ylim(-0.7, 2.7)
    ax_det.set_yticks([
        DETECTOR_Y["V1"],
        DETECTOR_Y["L1"],
        DETECTOR_Y["H1"],
    ])
    ax_det.set_yticklabels(["V1", "L1", "H1"])
    ax_det.set_ylabel("IFO")
    ax_det.set_xlabel(r"Offset from $T_0$ (s)")
    ax_det.grid(axis="x", alpha=0.3)
    ax_det.grid(axis="y", visible=False)

    ax_d90.set_xlim(window_min, window_max)
    ax_det.set_xlim(window_min, window_max)

    tick_positions, tick_labels = build_offset_xticks(
        detector_source_rows,
        summary,
        window_min,
        window_max,
    )
    
    ax_det.set_xticks(tick_positions)
    ax_det.set_xticklabels(tick_labels, rotation=0)

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.4,
            label=rf"$\mathrm{{NSBH}}\ (10,\,1.4)\,M_{{\odot}}$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.4,
            label=rf"$\mathrm{{BNS}}\ (1.4,\,1.4)\,M_{{\odot}}$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=r"$T_0$",
        ),
    ]

    ax_d90.legend(
        handles=family_handles,
        loc="best",
        frameon=True,
    )

    if summary is not None:
        input_t0 = summary.get("input_t0", summary.get("t0", ""))
        title = f"TDR summary around T0 = {input_t0}\nD90 shown for {iota_text}"
    else:
        title = f"TDR summary\nD90 shown for {iota_text}"
    
    ax_d90.set_title(title)

    if output_pdf is None:
        output_pdf = os.path.join(output_dir, "D90_and_IFO_windows_combined.pdf")

    fig.savefig(
        output_pdf,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Summary plot D90 inclination prior: {iota_text}")
    return output_pdf