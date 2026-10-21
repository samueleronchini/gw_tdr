Use the tool
============

Basic command
-------------

.. code-block:: bash

   tdr --output-dir <output_directory> --t0 <trigger_time> [options]


Required inputs
---------------

- ``--output-dir``: directory where results are written
- ``--t0``: trigger time (GPS seconds or ISO UTC)
- Localization input, one of:

  - ``--ra`` and ``--dec`` in degrees, or
  - ``--skymap-file`` with a HEALPix FITS sky map


Examples
--------

Run with sky position:

.. code-block:: bash

   tdr \
     --output-dir GRB_GIT \
     --t0 2020-03-26T12:24:47.903 \
     --ra 245.33 \
     --dec -21.08 \
   --snr-threshold 9 \
     --snr-type mf

Run with sky map:

.. code-block:: bash

   tdr \
     --output-dir GRB_GIT \
     --t0 2020-03-26T12:24:47.903 \
     --skymap-file examples/glg_healpix_all_bn200326517_v00.fit \
   --snr-threshold 9 \
     --snr-type mf


Command options
---------------

- ``--ra``: right ascension in degrees
- ``--dec``: declination in degrees
- ``--skymap-file``: optional localization sky map FITS file
- ``--iota-min`` and ``--iota-max``: custom inclination prior in degrees
- ``--snr-threshold``: detection threshold used for D90 (default: 9)
- ``--snr-type``: ``mf`` (matched-filter-like) or ``opt`` (optimal)
- ``--grb-name``: name for metadata and the combined plot (default: ``GRB``)
- ``--scan-start`` and ``--scan-end``: offsets from T0 in seconds (default: 0)
- ``--scan-step``: scan step in seconds (default: 32)

The inclination defaults are 0--45 degrees. The 0--90-degree prior is always
saved alongside a custom prior in the JSON files. Each targeted-range PDF
shows only the user's prior, with three solid curves for the mass configurations
and the inclination bounds in the legend title. Scan bins are processed
sequentially; the scan end is an explicit offset from T0, with no duration input.


Output overview
---------------

Typical output tree:

.. code-block:: text

   <output-dir>/
   |-- targ_range.log
   |-- scan_summary.json
   |-- <GRB-name>_D90_and_IFO_windows_combined.pdf
   |-- gwosc_cache/
   `-- scan_t0p0000/
       |-- ifos_used.txt
       |-- ifos_used.json
       |-- inj/
       |-- results/
       |-- psd_plot.pdf
       |-- bns_targeted_range.pdf
       |-- nsbh_targeted_range.pdf
       |-- results_bns.json
       |-- results_nsbh.json
       |-- range_map_bns_m1_1.4_m2_1.4.fits
       `-- range_map_bns_m1_1.4_m2_1.4.pdf

Each requested scan time gets its own directory. The combined plot uses the
mean network antenna magnitude over the map's 90% credible region for its
colors. Its legend is above the D90 panel, replacing the GRB title.
This area average is metadata and does not enter the D90 calculation.
D90 retains full-map probability-weighted sky sampling.

Python function
---------------

.. code-block:: python

   from targ_ac_git import run_grb

   summary = run_grb(
       grb_name="GRB200228291",
       t0="2020-02-28T06:58:33.806",
       output_dir="GRB_runs/GRB200228291",
       ra=333.89292,
       dec=-42.944167,
       scan_start=-1000,
       scan_end=3.584,
   )

The function returns scan settings, bin status, IFOs and antenna metadata.
The D90 values are saved in the per-bin result JSON files.
