# acq2h5 HDF5 File Format Specification

This document describes the exact on-disk structure of `.h5` files produced by
`acq2h5.py` (and by `mat_to_h5()` in `convert.py`) in the **acq-tools** project.
These files hold physiologic waveform data converted from BIOPAC AcqKnowledge
`.acq` recordings. The format is designed to be read identically from **Python
(`h5py`)** and **MATLAB (`h5read` / `h5readatt`)**.

> This is a *different* layout from the older `biopac_h5_format.md` reference.
> The two intentional differences: `timestamps_local` lives **inside** the
> `signals/` group, and timestamp values are **unix seconds** (not MATLAB
> datenums).

## Top-Level Structure

```
/
├── signals/                  (Group)
│   ├── <signal_name>         (Dataset: float64 1-D array; attrs: Fs, units)
│   ├── <signal_name>         ...
│   ├── ...
│   └── timestamps_local      (Dataset: float64 1-D array; attr: Fs)
├── event_markers             (Dataset: scalar string, JSON-encoded)
├── recording_start_utc       (Dataset: scalar string, ISO 8601 UTC)
├── recording_start_local     (Dataset: scalar string, local time)
└── Fs                        (Dataset: scalar float64 — shared sampling rate)
```

A real example file contains a `signals/` group with ~12 physiologic channels
(e.g. `abp`, `ecg`, `central_venous_pressure`, `etco2`, `heart_rate`, …) plus
`timestamps_local`, and the four top-level datasets.

---

## `/signals/` Group

Holds every per-sample 1-D array in the recording. Two kinds of dataset live
here: **physiologic channels** and the single **`timestamps_local`** vector.
All datasets in this group share the same length (one value per sample) and the
same sampling rate.

### Physiologic channel datasets

- **Path**: `/signals/<signal_name>` (e.g. `/signals/ecg`).
- **Name**: the channel's cleaned name — lowercase, alphanumeric + underscore,
  starting with a letter (e.g. `abp`, `internal_carotid_artery_flow`).
- **Data**: 1-D `float64` array of waveform samples. **The dataset *is* the
  waveform** — there is no nested `wave` field. Gaps between concatenated source
  files are filled with `NaN`.
- **Compression**: `gzip` (transparent to readers).

#### Attributes

| Attribute | Type             | Description                              |
|-----------|------------------|------------------------------------------|
| `Fs`      | scalar `float64` | Sampling frequency in Hz (e.g. `200.0`). |
| `units`   | UTF-8 string     | Unit of measurement (e.g. `"mmHg"`, `"Volts"`, `"mL/min"`). May be an empty string. |

Both attributes are always present on physiologic channels. `units` reflects
what was configured in BIOPAC and may be outdated; trust `Fs` over `units`.

### `timestamps_local` dataset

- **Path**: `/signals/timestamps_local`.
- **Data**: 1-D `float64` array, one value per sample, aligned element-for-element
  with every channel dataset (same length).
- **Content**: **unix timestamps** — seconds since the Unix epoch
  (1970-01-01 00:00:00 UTC), as floats. They encode local wall-clock time at the
  recording site (the epoch value already accounts for the local offset).
- **Attributes**: `Fs` only (scalar `float64`) — *no* `units`. The `Fs` attribute
  exists so a "for every dataset in `signals/`, read `Fs`" loop works uniformly.

> `timestamps_local` is deliberately stored *inside* `signals/` for convenience
> (one group holds every sample-aligned array). When you want "just the
> physiologic channels," iterate `signals/` and skip the dataset named
> `timestamps_local`.

---

## `/event_markers` Dataset

- **Type**: scalar string (UTF-8), JSON-encoded.
- **Content**: a JSON **object whose values are parallel arrays** (column-style),
  every array the same length = the number of markers. These are the BIOPAC
  comments/annotations placed during recording.

JSON keys (all present):

| Key                | Element type        | Description                                              |
|--------------------|---------------------|---------------------------------------------------------|
| `label`            | string              | Annotation text (e.g. `"Segment 1"`).                   |
| `sample_index`     | integer             | **1-indexed** sample position (MATLAB convention).      |
| `type_code`        | string              | Internal BIOPAC marker code (e.g. `"apnd"`).            |
| `type`             | string              | Human-readable marker type (e.g. `"Append"`, `"User Type 9"`). |
| `channel_number`   | number or `NaN`     | Associated channel number; `NaN` if not channel-bound.  |
| `channel`          | number or `NaN`     | Associated channel; `NaN` if not channel-bound.         |
| `seconds`          | number              | Marker time in seconds from recording start.            |
| `minutes`          | number              | Marker time in minutes from recording start.            |
| `time_local`       | string              | Local timestamp string `"YYYY-MM-DD HH:MM:SS.mmm"`.     |
| `date_created_utc` | string              | ISO 8601 UTC timestamp with offset.                     |

Notes:
- `sample_index` is **1-indexed** (it is meant to index MATLAB arrays directly).
  To index a 0-indexed Python/`h5py` array, subtract 1.
- `channel_number` / `channel` use the JSON token `NaN` for missing values
  (non-strict JSON, accepted by Python's `json.loads` and MATLAB's `jsondecode`).

---

## `/recording_start_utc` Dataset

- **Type**: scalar string (UTF-8).
- **Content**: ISO 8601 UTC start time of the recording, e.g.
  `"2025-10-22T11:41:54.243000+00:00"`. Taken from the earliest event marker.

## `/recording_start_local` Dataset

- **Type**: scalar string (UTF-8).
- **Content**: local-time start, formatted `"YYYY-MM-DD HH:MM:SS.mmm"`
  (e.g. `"2025-10-22 07:41:54.243"`).

## `/Fs` Dataset

- **Type**: scalar `float64`.
- **Content**: the single sampling frequency (Hz) shared by all channels. The
  converter validates that every channel has the same `Fs`, so this top-level
  scalar equals each channel's `Fs` attribute. Convenience accessor.

---

## Reading the File

### Python (h5py)

```python
import h5py, json
import numpy as np

with h5py.File('recording.h5', 'r') as f:
    # A waveform — the dataset IS the array
    ecg = f['signals/ecg'][:]                    # float64 ndarray
    fs    = f['signals/ecg'].attrs['Fs']         # 200.0
    units = f['signals/ecg'].attrs['units']      # 'Volts'  (str)

    # Per-sample timestamps (unix seconds)
    ts = f['signals/timestamps_local'][:]

    # All physiologic channel names (exclude the timestamp vector)
    channels = [k for k in f['signals'] if k != 'timestamps_local']

    # Event markers
    markers = json.loads(f['event_markers'][()])  # dict of parallel lists
    labels = markers['label']

    # Scalars (stored as bytes -> decode)
    start_utc = f['recording_start_utc'][()].decode('utf-8')
    sample_rate = float(f['Fs'][()])
```

Convert unix-second timestamps to datetimes:

```python
from datetime import datetime, timezone
dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]
```

### MATLAB

```matlab
ecg   = h5read('recording.h5', '/signals/ecg');            % waveform array
fs    = h5readatt('recording.h5', '/signals/ecg', 'Fs');    % 200.0
units = h5readatt('recording.h5', '/signals/ecg', 'units'); % 'Volts'

ts    = h5read('recording.h5', '/signals/timestamps_local');
markers = jsondecode(h5read('recording.h5', '/event_markers'));

% unix seconds -> datetime
t = datetime(ts, 'ConvertFrom', 'epochtime', 'TicksPerSecond', 1);
```

A ready-made loader, `loadh5.m`, reads the whole file into a struct (signals,
per-signal `Fs`/`units`, timestamps, and an event-marker table).

---

## Invariants (useful when validating or generating a file)

1. Every dataset under `/signals/` is 1-D `float64` and the **same length**.
2. Every `/signals/` dataset carries an `Fs` attribute; physiologic channels also
   carry `units`. `timestamps_local` carries `Fs` but **not** `units`.
3. `len(timestamps_local) == len(any channel)`.
4. Top-level `/Fs` equals every channel's `Fs` attribute (single shared rate).
5. `event_markers` decodes to an object of equal-length parallel arrays.
6. `sample_index` values are 1-indexed and lie in `[1, len(channel)]`.
7. Scalar string datasets (`event_markers`, `recording_start_utc`,
   `recording_start_local`) are stored as variable-length UTF-8; in Python they
   read back as `bytes` and should be `.decode('utf-8')`-ed.
