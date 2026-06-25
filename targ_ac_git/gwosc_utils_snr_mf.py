import os
import random
import socket
import subprocess
import time

import h5py
import numpy as np
from gwosc.locate import get_urls
from gwpy.timeseries import TimeSeries
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout, Timeout


class GWOSCTransientError(RuntimeError):
    """Temporary GWOSC/network/download failure.

    This should not be interpreted as the detector being offline. The trigger
    should be marked incomplete and rerun later.
    """


class GWOSCNoDataError(RuntimeError):
    """GWOSC query succeeded, but no usable strain data were found."""


def _is_transient_network_error(exc):
    """Return True for errors that are likely temporary network/API failures."""
    transient_types = (Timeout, ConnectTimeout, ReadTimeout, ConnectionError, socket.timeout, TimeoutError)

    if isinstance(exc, transient_types):
        return True

    msg = str(exc).lower()
    transient_markers = [
        "timed out",
        "timeout",
        "max retries exceeded",
        "connection reset",
        "connection aborted",
        "failed to establish a new connection",
        "temporary failure",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "remote end closed connection",
    ]

    return any(marker in msg for marker in transient_markers)


def _call_with_retries(func, *args, label="GWOSC call", max_tries=5, base_sleep=5.0, **kwargs):
    """Retry a GWOSC/network call when the failure appears transient."""
    last_exc = None

    for attempt in range(1, max_tries + 1):
        try:
            return func(*args, **kwargs)

        except Exception as exc:
            last_exc = exc

            if not _is_transient_network_error(exc):
                raise

            if attempt == max_tries:
                raise GWOSCTransientError(f"{label} failed after {max_tries} attempts: {exc}") from exc

            sleep_time = base_sleep * (2 ** (attempt - 1)) * random.uniform(0.7, 1.3)
            time.sleep(sleep_time)

    raise GWOSCTransientError(f"{label} failed: {last_exc}") from last_exc


GWOSC_SAMPLE_RATE = 4096
DEFAULT_SAMPLE_RATE = GWOSC_SAMPLE_RATE

# Virgo used a different public O3a strain release in the last two weeks.
O3A_LAST_TWO_WEEKS_START = 1253977218 - 14 * 86400


RUNS = [
    {
        "name": "O1",
        "start": 1126051217,
        "end": 1137254417,
        "ifos": {
            "H1": {
                "site": "LHO",
                "channel": "H1:DCS-CALIB_STRAIN_C02",
                "frame_type": "H1_HOFT_C02",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "L1": {
                "site": "LLO",
                "channel": "L1:DCS-CALIB_STRAIN_C02",
                "frame_type": "L1_HOFT_C02",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
        },
    },
    {
        "name": "O2",
        "start": 1164556817,
        "end": 1187733618,
        "ifos": {
            "H1": {
                "site": "LHO",
                "channel": "H1:DCH-CLEAN_STRAIN_C02",
                "frame_type": "H1_CLEANED_HOFT_C02",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "L1": {
                "site": "LLO",
                "channel": "L1:DCH-CLEAN_STRAIN_C02",
                "frame_type": "L1_CLEANED_HOFT_C02",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "V1": {
                "site": "Virgo",
                "channel": "V1:Hrec_hoft_V1O2Repro2A_16384Hz",
                "frame_type": "V1O2Repro2A",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
        },
    },
    {
        "name": "O3a",
        "start": 1238166018,
        "end": 1253977218,
        "ifos": {
            "H1": {
                "site": "LHO",
                "channel": "H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
                "frame_type": "H1_HOFT_CLEAN_SUB60HZ_C01",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "L1": {
                "site": "LLO",
                "channel": "L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
                "frame_type": "L1_HOFT_CLEAN_SUB60HZ_C01",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "V1": {
                "site": "Virgo",
                "sample_rate": GWOSC_SAMPLE_RATE,
                "variants": [
                    {
                        "start": 1238166018,
                        "end": O3A_LAST_TWO_WEEKS_START,
                        "channel": "V1:Hrec_hoft_16384Hz",
                        "frame_type": "V1Online",
                    },
                    {
                        "start": O3A_LAST_TWO_WEEKS_START,
                        "end": 1253977218,
                        "channel": "V1:Hrec_hoft_V1O3ARepro1A_16384Hz",
                        "frame_type": "V1O3Repro1A",
                    },
                ],
            },
        },
    },
    {
        "name": "O3b",
        "start": 1256655618,
        "end": 1269363618,
        "ifos": {
            "H1": {
                "site": "LHO",
                "channel": "H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
                "frame_type": "H1_HOFT_CLEAN_SUB60HZ_C01",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "L1": {
                "site": "LLO",
                "channel": "L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
                "frame_type": "L1_HOFT_CLEAN_SUB60HZ_C01",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "V1": {
                "site": "Virgo",
                "channel": "V1:Hrec_hoft_16384Hz",
                "frame_type": "V1Online",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
        },
    },
    {
        "name": "O4a",
        "start": 1368975618,
        "end": 1389456018,
        "ifos": {
            "H1": {
                "site": "LHO",
                "channel": "H1:GDS-CALIB_STRAIN_CLEAN_AR",
                "frame_type": "H1_HOFT_C00_AR",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "L1": {
                "site": "LLO",
                "channel": "L1:GDS-CALIB_STRAIN_CLEAN_AR",
                "frame_type": "L1_HOFT_C00_AR",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
        },
    },
    {
        "name": "O4b",
        "start": 1396796418,
        "end": 1422118818,
        "ifos": {
            "H1": {
                "site": "LHO",
                "channel": "H1:DCS-CALIB_STRAIN_CLEAN_AR01",
                "frame_type": "H1_HOFT_AR01",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "L1": {
                "site": "LLO",
                "channel": "L1:DCS-CALIB_STRAIN_CLEAN_AR01",
                "frame_type": "L1_HOFT_AR01",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
            "V1": {
                "site": "Virgo",
                "channel": "V1:Hrec_hoftRepro1AR_16384Hz",
                "frame_type": "HoftAR1U02",
                "sample_rate": GWOSC_SAMPLE_RATE,
            },
        },
    },
]


def _get_run_config(t0):
    for run in RUNS:
        if run["start"] <= t0 <= run["end"]:
            return run
    raise RuntimeError(f"t0={t0} is outside the ranges covered by public GWOSC runs")


def _resolve_ifo_config(run_cfg, ifo, t0):
    cfg = dict(run_cfg["ifos"][ifo])

    if "variants" not in cfg:
        return cfg

    for variant in cfg["variants"]:
        if variant["start"] <= t0 < variant["end"]:
            merged = dict(cfg)
            merged.update(variant)
            merged.pop("variants", None)
            return merged

    raise RuntimeError(f"No detector variant found for {ifo} in run {run_cfg['name']} at t0={t0}")


def _parse_hdf5_span(path):
    base = os.path.basename(path)
    parts = base.split("-")
    start = int(parts[-2])
    duration = int(parts[-1].split(".")[0])
    return start, duration


def _validate_cached_hdf5(path):
    try:
        with h5py.File(path, "r") as f:
            _ = f["strain"]["Strain"].shape
        return True
    except Exception:
        return False

def _acquire_file_lock(lock_path, stale_after=3600, poll_interval=0.2):
    """
    Acquire a simple inter-process lock using atomic file creation.

    This avoids multiple timebin workers downloading/writing the same GWOSC
    cache file at the same time.
    """
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"pid={os.getpid()}\ntime={time.time()}\n")
            return

        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > stale_after:
                    os.remove(lock_path)
                    continue
            except OSError:
                pass

            time.sleep(poll_interval)


def _release_file_lock(lock_path):
    try:
        os.remove(lock_path)
    except OSError:
        pass

 def _ensure_cached_file(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    filename = url.split("/")[-1]
    path = os.path.join(cache_dir, filename)

    # Lock per final cached file, so parallel timebins do not download/write
    # the same GWOSC file at the same time.
    lock_path = path + ".lock"

    # PID-specific temporary file avoids collisions between workers.
    tmp_path = f"{path}.pid{os.getpid()}.part"

    _acquire_file_lock(lock_path)

    try:
        if os.path.exists(path):
            if _validate_cached_hdf5(path):
                return path
            os.remove(path)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        import shutil

        print(
            f"Downloading {url} ... (this may take a while depending on your connection)",
            flush=True,
        )

        if shutil.which("wget"):
            cmd = [
                "wget",
                "--tries=5",
                "--timeout=60",
                "--waitretry=10",
                "-O",
                tmp_path,
                url,
            ]
        elif shutil.which("curl"):
            cmd = [
                "curl",
                "-L",
                "--progress-bar",
                "--retry",
                "5",
                "--retry-delay",
                "10",
                "--max-time",
                "600",
                "-o",
                tmp_path,
                url,
            ]
        else:
            raise GWOSCTransientError(
                "Neither wget nor curl is installed on this system"
            )

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise GWOSCTransientError(f"Download failed for {url}: {exc}") from exc

        if not os.path.exists(tmp_path):
            raise GWOSCTransientError(
                f"Download command did not create output file for {url}"
            )

        os.replace(tmp_path, path)

        if not _validate_cached_hdf5(path):
            try:
                os.remove(path)
            except OSError:
                pass
            raise GWOSCTransientError(f"Downloaded file is corrupted: {path}")

        return path

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        _release_file_lock(lock_path)

def _load_segment_from_cache(ifo, ifo_cfg, t_center, cache_dir):
    t_start = t_center - 128
    t_end = t_center + 128
    sample_rate = ifo_cfg.get("sample_rate", DEFAULT_SAMPLE_RATE)

    try:
        urls = _call_with_retries(
            get_urls,
            ifo,
            t_start,
            t_end,
            sample_rate=sample_rate,
            format="hdf5",
            label=f"get_urls {ifo} [{t_start}, {t_end}]",
            max_tries=5,
            base_sleep=5.0,
        )
    except GWOSCTransientError:
        raise
    except Exception as exc:
        if _is_transient_network_error(exc):
            raise GWOSCTransientError(f"Transient GWOSC get_urls failure for {ifo} in [{t_start}, {t_end}]: {exc}") from exc
        raise

    urls = urls or []
    if len(urls) == 0:
        raise GWOSCNoDataError(
            f"No GWOSC data for {ifo} in [{t_start}, {t_end}] "
            f"(channel={ifo_cfg['channel']}, frame={ifo_cfg['frame_type']})"
        )

    local_paths = [_ensure_cached_file(url, cache_dir) for url in urls]
    local_paths = sorted(local_paths, key=lambda path: _parse_hdf5_span(path)[0])

    n_expected = int(round((t_end - t_start) * sample_rate))
    full_data = np.full(n_expected, np.nan, dtype=np.float32)

    for path in local_paths:
        file_start, file_duration = _parse_hdf5_span(path)
        file_end = file_start + file_duration

        overlap_start = max(t_start, file_start)
        overlap_end = min(t_end, file_end)

        if overlap_end <= overlap_start:
            continue

        i0_file = int(round((overlap_start - file_start) * sample_rate))
        i1_file = int(round((overlap_end - file_start) * sample_rate))

        i0_out = int(round((overlap_start - t_start) * sample_rate))
        i1_out = i0_out + (i1_file - i0_file)

        with h5py.File(path, "r") as f:
            data = f["strain"]["Strain"][i0_file:i1_file].astype(np.float32)

        full_data[i0_out:i1_out] = data

    if np.isnan(full_data).any():
        n_missing = int(np.isnan(full_data).sum())
        raise GWOSCNoDataError(f"Incomplete coverage for {ifo}: {n_missing} samples missing in final segment")

    segment = TimeSeries(full_data, sample_rate=sample_rate, t0=t_start)
    return segment, local_paths


def _write_temp_segment_hdf5(segment, ifo, t_center, outdir):
    t_start = int(round(t_center - 128))
    duration = int(round(len(segment) / float(segment.sample_rate.value)))
    path = os.path.join(outdir, f"{ifo}-{ifo}_TMP-{t_start}-{duration}.hdf5")

    with h5py.File(path, "w") as f:
        group = f.create_group("strain")
        group.create_dataset("Strain", data=np.asarray(segment.value, dtype=np.float32))

    return path
