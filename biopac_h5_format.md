# BIOPAC HDF5 File Format Specification

This document describes the structure of `.h5` files created by `save_biopac_to_h5()` in `data-alignment/general_helper_functions.py`.

> **Note — `acq2h5.py` diverges from this reference.** The converter in this
> repository (`acq2h5.py`) produces a similar but intentionally different layout:
> - `timestamps_local` is stored **inside** the `/signals/` group (as
>   `/signals/timestamps_local`), not as a top-level dataset. It carries an `Fs`
>   attribute like the other signal datasets.
> - `timestamps_local` values are **unix timestamps** (float64 seconds since
>   1970-01-01 UTC), not MATLAB datenums. Convert in MATLAB with
>   `datetime(ts, 'ConvertFrom', 'epochtime', 'TicksPerSecond', 1)`.
> - Each signal dataset carries `Fs` and `units` as HDF5 attributes, and a
>   top-level scalar `/Fs` records the shared sampling rate.
>
> The rest of this document describes the original reference format.

## Top-Level Structure

```
/
├── signals/            (HDF5 Group)
│   ├── <signal_name>   (Dataset: float64 array, 1-D waveform)
│   ├── <signal_name>   ...
│   └── ...
├── event_markers       (Dataset: scalar string, JSON-encoded)
├── timestamps_local    (Dataset: float64 array, MATLAB datenums)
├── recording_start_utc   (Dataset: scalar string, UTC datetime)
└── recording_start_local (Dataset: scalar string, local datetime)
```

## `/signals/` Group

Each signal channel is stored as a dataset under the `signals` group. The dataset name is the signal's key from the source biopac_dict (e.g. `ecg`, `ca_flow`, `ecog_channel0`, `nirs_rSO2`).

### Dataset

- **Data**: 1-D `float64` NumPy array containing the waveform samples.

### Dataset Attributes

Each signal dataset may have the following HDF5 attributes:

| Attribute | Type    | Description                          |
|-----------|---------|--------------------------------------|
| `Fs`      | float64 | Sampling frequency in Hz (e.g. 200.0)|
| `units`   | string  | Unit of measurement (e.g. "mV")      |

Both attributes are optional (present only if the source dict contained `Fs` / `units` for that channel).

## `/event_markers` Dataset

- **Type**: Scalar string (UTF-8)
- **Content**: JSON-encoded list of event marker objects. Each object typically contains:
  - `"time (EST)"` - timestamp string for the event
  - Other fields depending on the BIOPAC acquisition setup (e.g. label, type)
- **To read**: `json.loads(f['event_markers'][()])` in Python

## `/timestamps_local` Dataset

- **Type**: 1-D `float64` array
- **Content**: MATLAB datenum values representing the local timestamp of each sample in the recording. These are days since January 0, 0000 (MATLAB convention).
- **Conversion to Python datetime**: Use `matlab_datenum_to_datetime()` from the same module, or:
  ```python
  from datetime import datetime, timedelta
  dt = datetime.fromordinal(int(datenum - 366)) + timedelta(days=float(datenum - 366) % 1)
  ```

## `/recording_start_utc` Dataset

- **Type**: Scalar string (UTF-8)
- **Content**: String representation of the recording start time in UTC.

## `/recording_start_local` Dataset

- **Type**: Scalar string (UTF-8)
- **Content**: String representation of the recording start time in local timezone.

## Notes

- All waveform signals are typically resampled to 200 Hz before saving, but check the `Fs` attribute on each dataset to confirm.
- The file is written with `h5py` using default compression settings (no compression).
- Optional top-level datasets (`event_markers`, `timestamps_local`, `recording_start_utc`, `recording_start_local`) are only present if the source biopac_dict contained those keys.
- Signal channels are identified by being dicts with a `'wave'` key in the source biopac_dict. All other recognized keys are stored as top-level datasets.
