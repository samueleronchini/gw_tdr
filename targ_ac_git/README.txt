TDR-GRB
=======

This repository contains a Python pipeline to estimate the Targeted Detectability Range (TDR) for compact-binary gravitational-wave signals associated with an external GRB trigger.

The pipeline retrieves public gravitational-wave strain data from the Gravitational Wave Open Science Center (GWOSC), estimates the detector power spectral density (PSD), computes compact-binary signal-to-noise ratios (SNRs), and evaluates the distance at which a chosen fraction of simulated sources would be detectable.

The current public version is designed for a single GRB trigger at a time.


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

The default inclination treatment compares two priors:

    0 deg < iota < 45 deg

and

    0 deg < iota < 90 deg

Alternatively, the user can provide a custom inclination interval using --iota-min and --iota-max.


INSTALLATION
============

Create a fresh conda environment:

    conda create -n TDR_grb python=3.11 pip
    conda activate TDR_grb

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

    TaylorT4

For NSBH systems, the waveform is:

    IMRPhenomNSBH

The NSBH calculations are performed for the SFHo and DD2 equations of state. The final public results_nsbh.json reports the EOS-averaged D90.


SNR STATISTIC
=============

The command-line option:

    --snr-statistic

accepts two values:

    mf
        matched-filter-like SNR

    opt
        optimal SNR

The default is:

    --snr-statistic mf

The SNR threshold used to define D90 is set with:

    --snr-threshold

The default value is:

    --snr-threshold 8.5


OUTPUT FILES
============

A typical output directory has the structure:

<output_dir>/
    targ_range.log
    ifos_used.txt
    ifos_used.json
    gwosc_cache/
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

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir <output_directory> \
        --t0 <trigger_time> \
        --ra <right_ascension_deg> \
        --dec <declination_deg> \
        --snr-threshold 8.5 \
        --snr-statistic mf

Example:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir GRB_GIT \
        --t0 2020-03-26T12:24:47.903 \
        --ra 245.33 \
        --dec -21.08 \
        --snr-threshold 8.5 \
        --snr-statistic mf


Example 2: using a sky map
--------------------------

Use this when a HEALPix localization file is available.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir <output_directory> \
        --t0 <trigger_time> \
        --skymap-file <skymap_file.fit> \
        --snr-threshold 8.5 \
        --snr-statistic mf

Example:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir GRB_GIT \
        --t0 2020-03-26T12:24:47.903 \
        --skymap-file glg_healpix_all_bn200326517_v00.fit \
        --snr-threshold 8.5 \
        --snr-statistic mf


Example 3: custom inclination interval
--------------------------------------

If --iota-min and --iota-max are omitted, the code evaluates both:

    0 deg < iota < 45 deg
    0 deg < iota < 90 deg

To run only a custom inclination range:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m targ_ac_git.targ_range_snr_mf \
        --output-dir GRB_GIT \
        --t0 2020-03-26T12:24:47.903 \
        --skymap-file glg_healpix_all_bn200326517_v00.fit \
        --iota-min 0 \
        --iota-max 30 \
        --snr-threshold 8.5 \
        --snr-statistic mf


NOTES
=====

The environment variables:

    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1

limit the number of CPU threads used internally by numerical libraries. This is useful on shared systems and avoids oversubscription.

The code currently performs a single-trigger analysis. It does not run a time scan, does not use multiprocessing, and does not read a CSV file of GRBs.

If a sky map is provided with --skymap-file, RA and Dec are not required. If no sky map is provided, both --ra and --dec must be given.