import argparse
import json
import logging
import os
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from astropy.time import Time

try:
    from .aux_snr_mf import compute_antennamap, compute_map, compute_range, map_samples, plot_final, inject, plot_psd, compute_localization_antenna_factor
    from .gwosc_utils_snr_mf import GWOSC_SAMPLE_RATE, GWOSCTransientError, GWOSCNoDataError, _get_run_config, _resolve_ifo_config, _load_segment_from_cache
    from .pipeline_utils_snr_mf import _make_psd_from_segment, _validate_psd_file
    from .summary_plot_snr_mf import plot_timebin_summary
except ImportError:
    from aux_snr_mf import compute_antennamap, compute_map, compute_range, map_samples, plot_final, inject, plot_psd, compute_localization_antenna_factor
    from gwosc_utils_snr_mf import GWOSC_SAMPLE_RATE, GWOSCTransientError, GWOSCNoDataError, _get_run_config, _resolve_ifo_config, _load_segment_from_cache
    from pipeline_utils_snr_mf import _make_psd_from_segment, _validate_psd_file
    from summary_plot_snr_mf import plot_timebin_summary


EOS_LIST = ["SFHo", "DD2"]

CHIRP_MASSES = {
    "bns": [(1, 1), (1.4, 1.4), (2.0, 2.0)],
    "nsbh": [(5, 1), (10, 1.4), (20, 2.0)],
}

MIN_T_BIN = 32.0
MAX_T_BIN = 256.0
MAX_TIMEBINS = 100

def setup_logger(log_file):
    logger = logging.getLogger("targ_range")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(log_file, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)

    return logger


def as_gps_seconds(t0):
    if "T" in str(t0):
        return Time(t0, format="isot", scale="utc").gps
    return float(t0)


def parse_snr_statistic(value):
    if value == "mf":
        return "matched_filter"
    if value == "opt":
        return "optimal"
    raise ValueError("snr_statistic must be either 'mf' or 'opt'")


def parse_iota_ranges(iota_min_deg, iota_max_deg):
    iota_min_deg = 0.0 if iota_min_deg is None else float(iota_min_deg)
    iota_max_deg = 45.0 if iota_max_deg is None else float(iota_max_deg)

    iota_min = np.radians(iota_min_deg)
    iota_max = np.radians(iota_max_deg)

    if iota_min < 0:
        raise ValueError("--iota-min must be >= 0 deg")
    if iota_max > np.pi / 2:
        raise ValueError("--iota-max must be <= 90 deg")
    if iota_min >= iota_max:
        raise ValueError("--iota-min must be smaller than --iota-max")

    return [
        {
            "label": f"{iota_min_deg:g}-{iota_max_deg:g}",
            "iota_min": iota_min,
            "iota_max": iota_max,
        }
    ]

def get_summary_iota_label(iota_ranges):
    """
    Return the inclination-prior label used by the pipeline.

    The summary plot must use the same inclination prior as the TDR calculation.
    """
    if len(iota_ranges) != 1:
        raise ValueError(
            "Expected exactly one inclination prior for the summary plot. "
            f"Got: {[r['label'] for r in iota_ranges]}"
        )

    return iota_ranges[0]["label"]
def _clean_offset(offset):
    """Return a stable numeric offset, avoiding -0.0."""
    offset = float(offset)
    if abs(offset) < 1e-9:
        return 0.0
    return round(offset, 6)


def build_time_offsets(t_start, t_end, t_bin):
    """
    Build offsets relative to T0.

    Always includes:
      - -t_start, if t_start > 0
      -  0
      - +t_end, if t_end > 0

    Also includes regular samples spaced by t_bin between the boundaries.
    """
    t_start = float(t_start)
    t_end = float(t_end)
    t_bin = float(t_bin)

    if t_start < 0:
        raise ValueError("--t-start must be >= 0")
    if t_end < 0:
        raise ValueError("--t-end must be >= 0")
    if t_bin <= 0:
        raise ValueError("--t-bin must be > 0")

    offsets = {0.0}

    if t_start > 0:
        offsets.add(_clean_offset(-t_start))

        current = t_bin
        while current < t_start:
            offsets.add(_clean_offset(-current))
            current += t_bin

    if t_end > 0:
        offsets.add(_clean_offset(t_end))

        current = t_bin
        while current < t_end:
            offsets.add(_clean_offset(current))
            current += t_bin

    return sorted(offsets)


def format_timebin_dir(offset):
    """
    Directory name for a time bin.

    Examples:
      offset = 0      -> scan_t0p0000
      offset = -32    -> scan_t0m0032
      offset = 128    -> scan_t0p0128
      offset = 12.5   -> scan_t0p0012p500
    """
    offset = _clean_offset(offset)

    if offset >= 0:
        sign = "p"
    else:
        sign = "m"

    value = abs(offset)

    if float(value).is_integer():
        value_text = f"{int(round(value)):04d}"
    else:
        value_text = f"{value:08.3f}".replace(".", "p")

    return f"scan_t0{sign}{value_text}"

def report_ifos_used(output_dir, ifo_status, strain_segments, online_ifos, logger):
    strain_available_ifos = sorted(strain_segments.keys())
    used_ifos = sorted(online_ifos)

    strain_text = ",".join(strain_available_ifos) if strain_available_ifos else "none"
    used_text = ",".join(used_ifos) if used_ifos else "none"

    message = f"IFOs with available strain data: {strain_text}\nIFOs used in the analysis:      {used_text}"

    print(message, flush=True)
    logger.info(message.replace("\n", " | "))

    with open(os.path.join(output_dir, "ifos_used.txt"), "w") as f:
        f.write(message + "\n")

    with open(os.path.join(output_dir, "ifos_used.json"), "w") as f:
        json.dump({"strain_available_ifos": strain_available_ifos, "used_ifos": used_ifos, "ifo_status": ifo_status}, f, indent=4)


def load_strain_segments(run_cfg, t_center, cache_dir, output_dir, logger):
    strain_segments = {}
    ifo_status = {}

    for ifo in run_cfg["ifos"]:
        ifo_status[ifo] = {"strain_available": False, "psd_ok": False, "usable": False, "network_failed": False, "reason": ""}

        try:
            ifo_cfg = _resolve_ifo_config(run_cfg, ifo, t_center)
            seg, used_paths = _load_segment_from_cache(ifo, ifo_cfg, t_center, cache_dir)

            strain_segments[ifo] = seg
            ifo_status[ifo]["strain_available"] = True

            logger.info(
                f"{os.path.basename(output_dir)}: run={run_cfg['name']} ifo={ifo} "
                f"channel={ifo_cfg['channel']} frame_type={ifo_cfg['frame_type']} "
                f"sample_rate={ifo_cfg['sample_rate']} strain usable from {used_paths}"
            )

        except GWOSCTransientError as e:
            ifo_status[ifo]["network_failed"] = True
            ifo_status[ifo]["reason"] = f"GWOSC/network failure: {e}"
            logger.info(f"{os.path.basename(output_dir)}: run={run_cfg['name']} ifo={ifo} GWOSC/network failure: {e}")

        except GWOSCNoDataError as e:
            ifo_status[ifo]["reason"] = f"no GWOSC data/incomplete coverage: {e}"
            logger.info(f"{os.path.basename(output_dir)}: run={run_cfg['name']} ifo={ifo} no usable data: {e}")

        except Exception as e:
            ifo_status[ifo]["reason"] = f"strain unavailable/incomplete: {e}"
            logger.info(f"{os.path.basename(output_dir)}: run={run_cfg['name']} ifo={ifo} unavailable: {e}")

    network_failed_ifos = [ifo for ifo, status in ifo_status.items() if status.get("network_failed", False)]
    if network_failed_ifos:
        msg = (
            f"{os.path.basename(output_dir)}: incomplete analysis because GWOSC/network failed for "
            f"IFOs={network_failed_ifos}. Loaded strain for IFOs={list(strain_segments.keys())}. "
            f"Rerun this trigger later; do not treat this as a real detector network."
        )
        with open(os.path.join(output_dir, "analysis_incomplete_network.txt"), "w") as f:
            f.write(msg + "\n")
            f.write(json.dumps(ifo_status, indent=4) + "\n")
        raise GWOSCTransientError(msg)

    if not strain_segments:
        msg = f"{os.path.basename(output_dir)}: no detectors available after successful GWOSC checks"
        with open(os.path.join(output_dir, "no_detectors_available.txt"), "w") as f:
            f.write(msg + "\n")
            f.write(json.dumps(ifo_status, indent=4) + "\n")
        logger.info(msg)

    return strain_segments, ifo_status


def build_psds(strain_segments, t_center, output_dir, ifo_status, logger):
    psd_list, psd_ifos, temp_files = [], [], []

    for ifo, seg in strain_segments.items():
        logger.info(f"{os.path.basename(output_dir)}: starting PSD for {ifo}")
        start = time.time()

        try:
            psd_obj, psd_path, temp_hdf5 = _make_psd_from_segment(ifo, seg, t_center, output_dir)
            _validate_psd_file(psd_path, ifo)

            psd_list.append(psd_obj)
            psd_ifos.append(ifo)
            temp_files.append(temp_hdf5)

            ifo_status[ifo]["psd_ok"] = True
            ifo_status[ifo]["usable"] = True
            ifo_status[ifo]["reason"] = "strain complete and PSD valid"

            logger.info(f"{os.path.basename(output_dir)}: finished PSD for {ifo} -> {psd_path} in {time.time() - start:.2f} s")

        except Exception as e:
            ifo_status[ifo]["psd_ok"] = False
            ifo_status[ifo]["usable"] = False
            ifo_status[ifo]["reason"] = f"PSD failed: {e}"
            logger.info(f"{os.path.basename(output_dir)}: PSD failed for {ifo}: {e} in {time.time() - start:.2f} s")

    return psd_list, sorted(psd_ifos), temp_files


def run_pycbc_optimal_snr(inj_file, res_file, online_ifos, output_dir):
    cmd = [
        "pycbc_optimal_snr",
        "--snr-columns",
        *[f"{ifo}:optimal_snr_{ifo}" for ifo in online_ifos],
        "--f-low", "30",
        "--seg-length", "256",
        "--sample-rate", "2048",
        "--input-file", inj_file,
        "--output-file", res_file,
    ]

    for ifo in online_ifos:
        cmd.extend(["--psd-file", f"{ifo}:{os.path.join(output_dir, f'{ifo.lower()}_psd.txt')}"])

    subprocess.run(cmd, check=True)


def create_injections_and_snr(t_center, online_ifos, output_dir, logger):
    inj_dir = os.path.join(output_dir, "inj")
    res_dir = os.path.join(output_dir, "results")
    os.makedirs(inj_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    ra_max, dec_max = compute_antennamap(online_ifos, t_center)
    logger.info(f"{os.path.basename(output_dir)}: starting injections and pycbc_optimal_snr")

    for cbc_type, masses in CHIRP_MASSES.items():
        for m1, m2 in masses:
            eos_values = EOS_LIST if cbc_type == "nsbh" else [None]

            for eos in eos_values:
                tag = f"_{eos}" if eos is not None else ""

                inject(cbc_type, t_center, [ra_max, dec_max], m1, m2, 0, output_dir, eos=eos if eos is not None else "SFHo")

                inj_file = os.path.join(inj_dir, f"injections_{cbc_type}{tag}_m1_{m1}_m2_{m2}.hdf")
                res_file = os.path.join(res_dir, f"results_{cbc_type}{tag}_m1_{m1}_m2_{m2}.hdf")

                if not os.path.exists(inj_file):
                    raise RuntimeError(f"Expected injection file was not created: {inj_file}")

                if os.path.exists(res_file):
                    continue

                start = time.time()
                logger.info(f"{os.path.basename(output_dir)}: starting pycbc_optimal_snr for {cbc_type}{tag} m1={m1} m2={m2}")

                run_pycbc_optimal_snr(inj_file, res_file, online_ifos, output_dir)

                logger.info(
                    f"{os.path.basename(output_dir)}: finished pycbc_optimal_snr "
                    f"for {cbc_type}{tag} m1={m1} m2={m2} in {time.time() - start:.2f} s"
                )

    return ra_max, dec_max


def choose_localization_samples(ra, dec, skymap_file, ra_max, dec_max):
    if skymap_file is not None:
        return (*map_samples(skymap_file), skymap_file)

    if ra is not None and dec is not None:
        return [ra], [dec], None

    return [ra_max], [dec_max], None


def run_single_trigger(t_center, output_dir, ra, dec, skymap_file, cache_dir, log_file, iota_ranges, snr_threshold, snr_statistic):
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(log_file)
    start_time = time.time()
    run_cfg = _get_run_config(t_center)

    print(f"[{os.path.basename(output_dir)}] START at t={t_center}", flush=True)

    ra = np.radians(float(ra)) if ra is not None else None
    dec = np.radians(float(dec)) if dec is not None else None

    seed = 12345 + int(round(t_center))
    np.random.seed(seed)
    random.seed(seed)

    strain_segments, ifo_status = load_strain_segments(run_cfg, t_center, cache_dir, output_dir, logger)
    if not strain_segments:
        return

    logger.info(f"{os.path.basename(output_dir)}: building PSDs for {list(strain_segments.keys())}")
    psd_list, online_ifos, temp_files = build_psds(strain_segments, t_center, output_dir, ifo_status, logger)

    report_ifos_used(output_dir, ifo_status, strain_segments, online_ifos, logger)
    logger.info(f"{os.path.basename(output_dir)}: final usable IFOs = {online_ifos}")
    logger.info(f"{os.path.basename(output_dir)}: IFO status = {ifo_status}")

    if not online_ifos:
        logger.info(f"{os.path.basename(output_dir)}: no usable detectors after strain+PSD checks")
        return

    logger.info(f"{os.path.basename(output_dir)}: starting plot_psd")
    plot_psd(psd_list, output_dir, ifos=online_ifos)
    logger.info(f"{os.path.basename(output_dir)}: finished plot_psd")

    ra_max, dec_max = create_injections_and_snr(t_center, online_ifos, output_dir, logger)

    map_iota_min = iota_ranges[0]["iota_min"]
    map_iota_max = iota_ranges[0]["iota_max"]

    logger.info(f"{os.path.basename(output_dir)}: starting compute_map")
    range_map = compute_map("bns", "1.4", "1.4", online_ifos, output_dir, map_iota_min, map_iota_max, snr_threshold, snr_statistic)
    logger.info(f"{os.path.basename(output_dir)}: finished compute_map")

    ra_samples, dec_samples, skymap = choose_localization_samples(ra, dec, skymap_file, ra_max, dec_max)
    
    antenna_mode = "max_over_grb_skymap_samples" if skymap_file is not None else "source_position"
    antenna_info = compute_localization_antenna_factor(
        online_ifos=online_ifos,
        t0=t_center,
        ra_samples=ra_samples,
        dec_samples=dec_samples,
        mode=antenna_mode,
    )
    
    logger.info(
        f"{os.path.basename(output_dir)}: antenna_factor={antenna_info['antenna_factor']} "
        f"mode={antenna_info['antenna_factor_mode']}"
    )
    
    logger.info(f"{os.path.basename(output_dir)}: starting compute_range")
    compute_range(
        ra_samples,
        dec_samples,
        online_ifos,
        CHIRP_MASSES,
        output_dir,
        iota_ranges,
        snr_threshold,
        snr_statistic,
        antenna_info=antenna_info,
    )
    logger.info(f"{os.path.basename(output_dir)}: finished compute_range")

    logger.info(f"{os.path.basename(output_dir)}: starting plot_final")
    plot_final(output_dir, range_map, skymap, [ra_samples, dec_samples], map_iota_min, map_iota_max)
    logger.info(f"{os.path.basename(output_dir)}: finished plot_final")
    logger.info(f"{os.path.basename(output_dir)}: completed analysis")

    print(f"[{os.path.basename(output_dir)}] DONE in {time.time() - start_time:.2f} s ({(time.time() - start_time) / 60:.2f} min)", flush=True)

    for tmp in temp_files:
        try:
            os.remove(tmp)
        except OSError:
            pass

def run_timebin_job(job):
    """
    Worker for one time bin.

    One process handles one time bin. Outputs are written into a dedicated
    scan_t0mXXXX / scan_t0pXXXX directory.
    """
    offset = job["offset"]
    t_center = job["t0"] + offset
    output_dir = os.path.join(job["base_output_dir"], format_timebin_dir(offset))
    log_file = os.path.join(output_dir, "targ_range.log")

    os.makedirs(output_dir, exist_ok=True)

    try:
        run_single_trigger(
            t_center=t_center,
            output_dir=output_dir,
            ra=job["ra"],
            dec=job["dec"],
            skymap_file=job["skymap_file"],
            cache_dir=job["cache_dir"],
            log_file=log_file,
            iota_ranges=job["iota_ranges"],
            snr_threshold=job["snr_threshold"],
            snr_statistic=job["snr_statistic"],
        )

        return {
            "offset": offset,
            "t_center": t_center,
            "output_dir": output_dir,
            "status": "done",
            "error": None,
        }

    except Exception as exc:
        return {
            "offset": offset,
            "t_center": t_center,
            "output_dir": output_dir,
            "status": "failed",
            "error": str(exc),
        }

def build_parser():
    parser = argparse.ArgumentParser(description="Compute targeted detectability ranges for a single GRB trigger.")

    parser.add_argument("--output-dir", required=True, help="Directory where all outputs will be written.")
    parser.add_argument("--t0", required=True, help="Trigger time as GPS seconds or ISO UTC, e.g. 2020-03-26T12:24:47.903.")
    parser.add_argument("--t-start", type=float,  default=0.0, help="Positive time before T0, in seconds, from which to start the TDR scan. Default: 0.")
        
    parser.add_argument("--t-end", type=float,  default=0.0, help="Positive time after T0, in seconds, at which to end the TDR scan. Default: 0.")
        
    parser.add_argument(
        "--t-bin",
        type=float,
        default=32.0,
        help=(
            "Timebin spacing in seconds. "
            "Must be between 32 and 256 s. "
            "The total number of generated time bins must not exceed 100."
        ),
    )
        
    parser.add_argument("--n-cpus", type=int, default=1, help="Number of CPUs to use. One CPU is used per time bin, up to the number of available time bins. Default: 1.")
    parser.add_argument("--ra", type=float, default=None, help="Right ascension in degrees.")
    parser.add_argument("--dec", type=float, default=None, help="Declination in degrees.")
    parser.add_argument("--skymap-file", type=str, default=None, help="Optional HEALPix sky map FITS file.")

    parser.add_argument("--iota-min", type=float, default=0.0, help="Minimum inclination angle in degrees.")
    parser.add_argument("--iota-max", type=float, default=45.0, help="Maximum inclination angle in degrees.")
    parser.add_argument("--snr-threshold", type=float, default=8.5, help="SNR threshold used to define D90. Default: 8.5.")
    parser.add_argument("--snr-statistic", choices=["mf", "opt"], default="mf",
        help="SNR statistic used to define the TDR: 'mf' for matched-filter SNR, 'opt' for optimal SNR. Default: mf.",
    )

    return parser


def validate_timebin_arguments(args, parser):
    """
    Validate user-provided multi-timebin settings before running the analysis.
    """
    if args.t_start < 0:
        parser.error("--t-start must be >= 0 s")

    if args.t_end < 0:
        parser.error("--t-end must be >= 0 s")

    if args.t_bin < MIN_T_BIN or args.t_bin > MAX_T_BIN:
        parser.error(
            f"--t-bin must be between {MIN_T_BIN:g} and {MAX_T_BIN:g} s. "
            f"Received --t-bin {args.t_bin:g} s."
        )
        
def targ_range(args=None):
    parser = build_parser()

    if args is None:
        args = parser.parse_args()
    elif isinstance(args, dict):
        args = argparse.Namespace(**args)

    if args.skymap_file == "None":
        args.skymap_file = None

    if args.skymap_file is None and (args.ra is None or args.dec is None):
        raise ValueError("Please provide either --skymap-file or both --ra and --dec.")

    validate_timebin_arguments(args, parser)

    snr_threshold = float(args.snr_threshold)
    if snr_threshold <= 0:
        raise ValueError("--snr-threshold must be positive")

    snr_statistic = parse_snr_statistic(args.snr_statistic)
    iota_ranges = parse_iota_ranges(args.iota_min, args.iota_max)
    summary_iota_label = get_summary_iota_label(iota_ranges)
    
    start_time = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "targ_range.log")
    open(log_file, "w").close()

    logger = setup_logger(log_file)
    t0 = as_gps_seconds(args.t0)
    event_run_cfg = _get_run_config(t0)
    t_start = float(args.t_start)
    t_end = float(args.t_end)
    t_bin = float(args.t_bin)
    
    time_offsets = build_time_offsets(t_start, t_end, t_bin)

    if len(time_offsets) > MAX_TIMEBINS:
        parser.error(
            f"The requested configuration generates {len(time_offsets)} time bins, "
            f"which exceeds the maximum allowed value of {MAX_TIMEBINS}. "
            "Please decrease --t-start and/or --t-end, or increase --t-bin "
            "to reduce the time-bin sampling frequency."
        )
    if args.n_cpus < 1:
        raise ValueError("--n-cpus must be >= 1")
    
    n_workers = min(int(args.n_cpus), len(time_offsets))
    logger.info(f"Starting targ_range for output_dir={args.output_dir}")
    logger.info(f"Input t0={args.t0}")
    logger.info(f"Input ra={args.ra}, dec={args.dec}")
    logger.info(f"Input skymap_file={args.skymap_file}")
    logger.info(f"Input snr_statistic={snr_statistic}")
    logger.info(f"Input snr_threshold={snr_threshold}")
    logger.info(f"Input iota_ranges={iota_ranges}")

    print(f"EVENT RUN = {event_run_cfg['name']}", flush=True)
    print(f"USING GWOSC STRAIN SAMPLE RATE: {GWOSC_SAMPLE_RATE} Hz", flush=True)
    print(f"USING SNR STATISTIC: {snr_statistic}", flush=True)
    print(f"USING SNR THRESHOLD: {snr_threshold:g}", flush=True)
    print("USING INCLINATION PRIOR(S):", flush=True)

    for prior in iota_ranges:
        print(f"  {np.degrees(prior['iota_min']):.1f} deg <= iota <= {np.degrees(prior['iota_max']):.1f} deg", flush=True)

    cache_dir = os.path.join(args.output_dir, "gwosc_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    print("RUNNING MULTI-TIMEBIN ANALYSIS", flush=True)
    print(f"NUMBER OF TIME BINS: {len(time_offsets)}", flush=True)
    print(f"USING CPU WORKERS: {n_workers}", flush=True)
    print("TIME OFFSETS RELATIVE TO T0:", flush=True)
    for offset in time_offsets:
        print(f"  {offset:+g} s -> {format_timebin_dir(offset)}", flush=True)
    
    jobs = [
        {
            "offset": offset,
            "t0": t0,
            "base_output_dir": args.output_dir,
            "ra": args.ra,
            "dec": args.dec,
            "skymap_file": args.skymap_file,
            "cache_dir": cache_dir,
            "iota_ranges": iota_ranges,
            "snr_threshold": snr_threshold,
            "snr_statistic": snr_statistic,
        }
        for offset in time_offsets
    ]
    
    results = []
    
    try:
        if n_workers == 1:
            for job in jobs:
                result = run_timebin_job(job)
                results.append(result)
    
                if result["status"] == "failed":
                    print(f"[{format_timebin_dir(result['offset'])}] FAILED: {result['error']}", flush=True)
                else:
                    print(f"[{format_timebin_dir(result['offset'])}] DONE", flush=True)
    
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_offset = {
                    executor.submit(run_timebin_job, job): job["offset"]
                    for job in jobs
                }
    
                for future in as_completed(future_to_offset):
                    result = future.result()
                    results.append(result)
    
                    if result["status"] == "failed":
                        print(f"[{format_timebin_dir(result['offset'])}] FAILED: {result['error']}", flush=True)
                    else:
                        print(f"[{format_timebin_dir(result['offset'])}] DONE", flush=True)
    
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        with open(os.path.join(args.output_dir, "analysis_failed.txt"), "w") as f:
            f.write(str(e) + "\n")
        raise
    
    results = sorted(results, key=lambda item: item["offset"])
    
    summary_path = os.path.join(args.output_dir, "multi_timebin_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "t0": t0,
                "input_t0": args.t0,
                "t_start": t_start,
                "t_end": t_end,
                "t_bin": t_bin,
                "n_timebins": len(time_offsets),
                "n_workers": n_workers,
                "results": results,
            },
            f,
            indent=4,
        )
    
    failed = [item for item in results if item["status"] == "failed"]
    
    if failed:
        failed_path = os.path.join(args.output_dir, "analysis_failed.txt")
        with open(failed_path, "w") as f:
            for item in failed:
                f.write(f"{format_timebin_dir(item['offset'])}: {item['error']}\n")
    
        raise RuntimeError(f"{len(failed)} time bin(s) failed. See {failed_path}")
    
    print(f"MULTI-TIMEBIN SUMMARY: {summary_path}", flush=True)

    # Summary Plot for D90
    summary_iota_label = iota_ranges[0]["label"]
    summary_plot_path = plot_timebin_summary(
        output_dir=args.output_dir,
        iota_label=summary_iota_label,
    )

    print(f"SUMMARY PLOT: {summary_plot_path}", flush=True)
    
    elapsed = time.time() - start_time
    print(f"ANALYSIS COMPLETE in {elapsed:.2f} s ({elapsed / 60:.2f} min)", flush=True)


if __name__ == "__main__":
    targ_range()
