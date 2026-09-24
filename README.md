# Kp index forecasting

Research code maintained by Eric Chen for estimating the geomagnetic Kp index from solar-wind drivers at UC San Diego. The project compares a physical linear baseline with temporal models and transformer experiments, with particular attention to storm detection and false alarms.

This is a selected source snapshot from September 24, 2026. It includes the core training pipeline and representative temporal and transformer experiments. Research data, trained models, generated results, manuscripts, and later exploratory studies are not bundled.

## Forecast interpretation

An upstream system supplies forecast solar-wind fields at a target time T. This code estimates Kp at that same time T using magnetic-field components, density, velocity, and derived features. Forecast lead time comes from the upstream driver forecast.

Experiments using historical observed drivers measure conditional driver-to-Kp performance. They do not establish end-to-end forecast skill with uncertain future drivers. Storm evaluation must consider precision, recall, and false-alarm rate together; high recall alone is insufficient.

## Code map

| Files | Purpose |
| --- | --- |
| `train_kp.py`, `cli.py` | Training and evaluation command line |
| `data_acquisition.py`, `read_module.py`, `data_filter.py` | Read, align, and filter input data |
| `run_kp_model.py`, `model_training.py`, `parabolic_fit.py` | Physical baseline, coefficient fitting, and Bz-dip terms |
| `temporal_model.py` | Temporal features, chronological evaluation, storm diagnostics, and model persistence |
| `unified_refresh_pipeline.py` | Windowed forecasting comparisons and shared experiment utilities |
| `experiment16_transformer_seq2seq.py` | Sequence construction and comparison experiment |
| `experiment17_pytorch_transformer.py` | PyTorch transformer experiment |
| `test_*.py` | Existing synthetic-data tests for the included experiments |

## Setup

Use a separate Python environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_kp.py --help
```

For the PyTorch experiment, also install `torch`. Some optional temporal estimators use `lightgbm` or `xgboost`. Dependencies are not pinned to a reproduced research environment in this snapshot.

## Inputs

The original readers expect annual files named `Bxyz_insitu_e3_YYYY.txt`, `e3_YYYY.txt`, and `kp_YYYY.txt`. Supply their directories with `-bd`, `-e3d`, and `-kpd`. The exact column parsing is in `read_module.py`; these are project-specific text formats, not arbitrary NOAA downloads.

A previously prepared dataframe can instead be supplied with `-dfr`. Core fields include `year`, `doy`, `hour`, `bx`, `by`, `bz`, `density`, `velocity`, and the observed `kp` target. See `read_module.reload_df` and `data_acquisition.acquire_reload` for the expected saved-dataframe format.

The transformer scripts default to a local prepared dataframe at `testing/2008_2025_df.txt`; override it with `KP_EXPERIMENT16_DATA` or `KP_EXPERIMENT17_DATA`. This file is not distributed here.

Run the included tests after installing the dependencies:

```sh
python -m unittest discover -p 'test_*.py'
```

This snapshot has been checked for Python syntax and command-line argument parsing. Full training and end-to-end reproduction require the external data and scientific dependencies.

## Attribution and license

The original `train_kp` code was written by Benjamin Pieczynski for UCSD A&A Solar Labs. This repository contains that baseline together with subsequent Kp research code from Eric Chen's working project. Original author notices are retained in the source files.

The original MIT license, copyright 2024 UCSD A&A Solar Labs, is preserved in [license.txt](license.txt).
