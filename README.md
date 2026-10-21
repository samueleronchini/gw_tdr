GW Targetet Detectability Range
===============================

This repository contains a Python pipeline to estimate the Targeted Detectability Range (TDR) for compact-binary gravitational-wave signals associated with an external GRB trigger.

The pipeline retrieves public gravitational-wave strain data from the Gravitational Wave Open Science Center (GWOSC), estimates the detector power spectral density (PSD), computes compact-binary signal-to-noise ratios (SNRs), and evaluates the distance at which a chosen fraction of simulated sources would be detectable.

The pipeline analyzes one GRB at a time, either at T0 or across a user-selected time interval. It provides both the installed `tdr` command and a callable Python function, `run_grb()`.


OVERVIEW
========

For a given trigger time, sky position or sky map, and inclination-angle prior, the pipeline:

1. identifies the relevant LVK observing run;
2. determines which public GWOSC strain data are available for each detector;
3. downloads a 256 s strain segment around the trigger time;
4. estimates the detector PSD using Welch averaging;
5. creates BNS and NSBH signal injections using PyCBC;
6. computes the optimal SNR using pycbc_optimal_snr;
7. optionally converts optimal SNR to a matched-filter-like SNR;
8. estimates the targeted detectability range, D90;
9. produces TDR plots, sky maps, PSD plots, and JSON result files.

The default calculation saves results for two inclination priors:

    0 deg < iota < 45 deg

and

    0 deg < iota < 90 deg

Use --iota-min and --iota-max for a custom inclination interval. The JSON files always include the 0--90-degree prior alongside the user's prior. Each targeted-range PDF shows only the user's prior, with three solid curves for the three mass configurations. Inclination samples are uniform in cos(iota).


INSTALLATION
============

**Prerequisites:** You must have either `wget` or `curl` installed on your system to download the strain data from GWOSC.

Create a fresh conda environment:

    conda create -n TDR python=3.11 pip
    conda activate TDR

Clone the repository:

    git clone <repo-url>
    cd <repo-name>

Install the required packages:

    pip install -r requirements.txt

Install the repository in editable mode:

    pip install -e .

Test the installation:

    python -m targ_ac_git.targ_range_snr_mf --help


REPOSITORY STRUCTURE
====================

<repo-name>/
    README.txt
    requirements.txt
    pyproject.toml
    targ_ac_git/
        __init__.py
        targ_range_snr_mf.py
        aux_snr_mf.py
        gwosc_utils_snr_mf.py
        pipeline_utils_snr_mf.py

Main files:

    targ_range_snr_mf.py
        Main command-line interface and pipeline driver.

    gwosc_utils_snr_mf.py
        GWOSC run configuration and strain-data retrieval.

    pipeline_utils_snr_mf.py
        PSD estimation utilities.

    aux_snr_mf.py
        Injections, SNR conversion, TDR calculation, and plotting.


STRAIN DATA
===========

The pipeline uses public calibrated strain data from GWOSC.

The observing-run configuration is defined in:

    targ_ac_git/gwosc_utils_snr_mf.py

The code selects the correct strain release and detector configuration based on the input trigger time.

Examples of configured public strain channels include:

    O1:
        H1:DCS-CALIB_STRAIN_C02
        L1:DCS-CALIB_STRAIN_C02

    O2:
        H1:DCH-CLEAN_STRAIN_C02
        L1:DCH-CLEAN_STRAIN_C02
        V1:Hrec_hoft_V1O2Repro2A_16384Hz

    O3:
        H1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01
        L1:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01
        V1 run-dependent public strain channels

    O4a:
        H1:GDS-CALIB_STRAIN_CLEAN_AR
        L1:GDS-CALIB_STRAIN_CLEAN_AR

    O4b:
        H1:DCS-CALIB_STRAIN_CLEAN_AR01
        L1:DCS-CALIB_STRAIN_CLEAN_AR01
        V1:Hrec_hoftRepro1AR_16384Hz

For a trigger time t0, the pipeline downloads the interval:

    t0 - 128 s <= t <= t0 + 128 s

GWOSC files are located using:

    from gwosc.locate import get_urls

The downloaded files are saved in:

    <output_dir>/gwosc_cache/

The cache is kept after the run for reproducibility and debugging.


PSD ESTIMATION
==============

For each available detector, the pipeline stitches the required 256 s strain segment and estimates the PSD using Welch averaging.

The default PSD configuration is:

    PSD segment length: 16 s
    PSD overlap:        8 s

The PSD files are written as two-column text files:

    h1_psd.txt
    l1_psd.txt
    v1_psd.txt

These PSDs are passed to pycbc_optimal_snr.


COMPACT-BINARY SYSTEMS
======================

The pipeline currently evaluates the following mass combinations.

BNS:

    m1 = 1.0 Msun,  m2 = 1.0 Msun
    m1 = 1.4 Msun,  m2 = 1.4 Msun
    m1 = 2.0 Msun,  m2 = 2.0 Msun

NSBH:

    mBH = 5.0 Msun,   mNS = 1.0 Msun
    mBH = 10.0 Msun,  mNS = 1.4 Msun
    mBH = 20.0 Msun,  mNS = 2.0 Msun

For BNS systems, the waveform is:

    TaylorF2

For NSBH systems, the waveform is:

    IMRPhenomNSBH

The NSBH calculations are performed for the SFHo and DD2 equations of state. The final public results_nsbh.json reports the EOS-averaged D90.


SNR TYPE
=============

The command-line option:

    --snr-type

accepts two values:

    mf
        matched-filter-like SNR

    opt
        optimal SNR

The default is:

    --snr-type mf

The SNR threshold used to define D90 is set with:

    --snr-threshold

The default value is:

    --snr-threshold 9


OUTPUT FILES
============

A typical output directory has the structure:

```text
<output_dir>/
    targ_range.log
    scan_summary.json
    <GRB-name>_D90_and_IFO_windows_combined.pdf
    gwosc_cache/
    scan_t0p0000/
        ifos_used.txt
        ifos_used.json
        inj/
        results/
        h1_psd.txt
        l1_psd.txt
        v1_psd.txt
        psd_plot.pdf
        bns_targeted_range.pdf
        nsbh_targeted_range.pdf
        range_map_bns_m1_1.4_m2_1.4.fits
        range_map_bns_m1_1.4_m2_1.4.pdf
        results_bns.json
        results_nsbh.json
```

Each requested scan time has its own bin directory. Detector-specific files exist for the detectors used in that bin. The range map uses the user's inclination bounds and retains all three distance contours. Each targeted-range PDF shows three solid mass curves for the user's prior, with the inclination bounds in the legend title. The result JSON files include both the user's prior and the 0--90-degree prior. The evolution plot shows BNS (1.4, 1.4) and NSBH (10, 1.4) D90 for the user prior with an IFO subplot. Its legend is above the D90 panel, in place of the GRB title.

The file:

    ifos_used.txt

summarizes which detector strain data were available and which detectors were used in the analysis.

The files:

    results_bns.json
    results_nsbh.json

contain the final D90 values.


RUNNING THE PIPELINE
====================

Example 1: fixed sky position
-----------------------------

Use this when the GRB sky position is known.

    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir <output_directory> \
        --t0 <trigger_time> \
        --ra <right_ascension_deg> \
        --dec <declination_deg> \
        --snr-threshold 9 \
        --snr-type mf

Example:

    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir GRB_GIT \
        --t0 2020-03-26T12:24:47.903 \
        --ra 245.33 \
        --dec -21.08 \
        --snr-threshold 9 \
        --snr-type mf


Example 2: using a sky map
--------------------------

Use this when a HEALPix localization file is available.

    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir <output_directory> \
        --t0 <trigger_time> \
        --skymap-file <skymap_file.fit> \
        --snr-threshold 9 \
        --snr-type mf

Example:

    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir GRB_GIT \
        --t0 2020-03-26T12:24:47.903 \
        --skymap-file examples/glg_healpix_all_bn200326517_v00.fit \
        --snr-threshold 9 \
        --snr-type mf


Example 3: custom inclination interval
--------------------------------------

If --iota-min and --iota-max are omitted, the code evaluates both:

    0 deg < iota < 45 deg
    0 deg < iota < 90 deg

To plot a custom inclination range and save both priors in JSON:

    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir GRB_GIT \
        --t0 2020-03-26T12:24:47.903 \
        --skymap-file examples/glg_healpix_all_bn200326517_v00.fit \
        --iota-min 0 \
        --iota-max 30 \
        --snr-threshold 9 \
        --snr-type mf


NOTES
=====


The code analyzes a single GRB, with sequential scan bins and no CPU-count argument or source CSV. The default scan start and end are both zero, so the default analysis runs at T0 only. Detector downloads may run concurrently.

If a sky map is provided with --skymap-file, RA and Dec are not required. If no sky map is provided, both --ra and --dec must be given.


PYTHON FUNCTION AND TIME SCANS
=============================

After installing this version of TDR, call the function from Python or a notebook:

```python
from targ_ac_git import run_grb

summary = run_grb(
    grb_name="GRB200228291",
    t0="2020-02-28T06:58:33.806",
    output_dir="GRB_runs/GRB200228291",
    ra=333.89292,
    dec=-42.944167,
    skymap_file="/absolute/path/to/localization.fit",
    scan_start=-1000,
    scan_end=3.584,
)
```

The function is implemented in `targ_ac_git/targ_range_snr_mf.py` and is also importable from that module. It returns the scan summary dictionary, including bin status, detectors and antenna metadata. D90 values are saved in each bin's `results_bns.json` and `results_nsbh.json`. Failed bins are recorded and cause an exception after the scan.

| Setting | Default |
| --- | --- |
| Scan start relative to T0 | 0 s |
| Scan end relative to T0 | 0 s |
| Scan step | 32 s |
| Minimum inclination | 0 degrees |
| Maximum inclination | 45 degrees |
| SNR threshold | 9 |
| SNR type | `mf` |

The same inputs are available through the existing installed command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
tdr \
    --grb-name GRB200228291 \
    --output-dir GRB_runs/GRB200228291 \
    --t0 2020-02-28T06:58:33.806 \
    --ra 333.89292 \
    --dec=-42.944167 \
    --skymap-file /absolute/path/to/localization.fit \
    --scan-start=-1000 \
    --scan-end 3.584 \
    --scan-step 32 \
    --iota-min 0 \
    --iota-max 45 \
    --snr-threshold 9
```

The localization FITS file must exist locally. Paths are resolved relative to the current working directory unless absolute paths are provided. `--grb-name` defaults to `GRB` for existing command-line callers such as the web app; the Python function requires a name. The installed command remains `tdr`. Result files live in scan-bin subdirectories, including for T0-only runs.

The scan includes its exact endpoints and the T0-anchored step grid within those limits. T0 is included when it lies within the interval. Set `--scan-end` explicitly in seconds relative to T0; no duration or D90 input is needed.

ANTENNA METADATA
================

With a localization map, the evolution plot and `antenna_factor.network` in the JSON use the arithmetic mean of the network antenna magnitude over equal-area native HEALPix pixel centers in the map's 90% highest-probability region, including the crossing pixel and boundary ties:

$$
\overline{F}_{\mathrm{net}} = \frac{1}{N_{90}}\sum_{p\in R_{90}}\sqrt{\sum_d(F_{+,d,p}^2+F_{\times,d,p}^2)}.
$$

This is an area average, without localization-probability or PSD weighting. Per-detector mean magnitudes are stored as `antenna_factor.per_ifo[ifo].F_mean`. For a fixed position, the plotting factor is evaluated at that source position.

This plotting statistic does not enter D90. The original D90 calculation samples the full localization probability and evaluates per-detector effective distances for the sampled sky positions and orientations. The all-sky injection-reference response is stored separately as `antenna_factor.injection_reference` and is rescaled through the effective-distance ratios.

Recreate the evolution plot from existing output:

```bash
python -m targ_ac_git.plot_time_evolution --output-dir GRB_runs/GRB200228291
```

This uses the existing saved antenna metadata; it does not recompute that metadata.

CHECKS
======

Install `pytest` in the development environment, then run:

```bash
python -m pytest tests -q
```

The numerical-function regression check uses source hashes from the supplied baseline. The API tests exercise scan defaults, return values, legacy CLI arguments, failure reporting and the regional mean. The optional waveform test uses synthetic strain with real PyCBC waveforms and PSDs:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
RUN_GRB_INTEGRATION=1 python -m pytest tests/test_grb_integration.py -q
```
