# acq2h5 HDF5 File Format Specification

This document describes the exact on-disk structure of `.h5` files produced by
`acq2h5.py` (and by `mat_to_h5()` in `convert.py`) in the **acq-tools** project.
These files hold physiologic waveform data converted from BIOPAC AcqKnowledge
`.acq` recordings. The format is designed to be read identically from **Python
(`h5py`)** and **MATLAB (`h5read` / `h5readatt`)**.

> **For a reader new to these files:** the structure is flat and self-describing.
> A `.h5` file is one recording (or several `.acq` recordings concatenated in
> time, with `NaN`-filled gaps between them). Every per-sample array — each
> waveform plus the shared time axis — lives under `/signals/` and has the same
> length. `timestamps_unix` is the alignment key: it gives the absolute wall-clock
> instant of every sample. Everything else (`event_markers`, the
> `recording_start_*` strings, top-level `Fs`) is metadata at the root.

> This is a *different* layout from the older `biopac_h5_format.md` reference.
> The two intentional differences: `timestamps_unix` lives **inside** the
> `signals/` group, and timestamp values are **unix seconds** (not MATLAB
> datenums).

> **Relationship to the `.mat` format.** The sibling `.mat` files (from
> `acq2mat.py`) hold the *same* data, but the per-sample timestamp vector is
> named **`timestamps_local`** there, not `timestamps_unix`. The values are
> identical unix epoch seconds — only the field name differs. `convert.py`
> (`mat_to_h5` / `h5_to_mat`) maps between the two names. If you write your own
> converter, remember this rename: a `.mat`→`.h5` step must treat
> `timestamps_local` as the timestamp axis, not as a physiologic channel.

## Top-Level Structure

```
/                             (root; attrs: file_revision, native_samples_per_second)
├── signals/                  (Group)
│   ├── <signal_name>         (Dataset: float64 1-D array; attrs: Fs, units,
│   │                          order_num, point_count, frequency_divider,
│   │                          raw_scale_factor, raw_offset, dtype)
│   ├── <signal_name>         ...
│   ├── ...
│   └── timestamps_unix       (Dataset: float64 1-D array; attr: Fs)
├── event_markers             (Dataset: scalar string, JSON-encoded)
├── recording_start_utc       (Dataset: scalar string, ISO 8601 UTC)
├── recording_start_local     (Dataset: scalar string, local time)
└── Fs                        (Dataset: scalar float64 — shared sampling rate)
```

A real example file contains a `signals/` group with ~12 physiologic channels
(e.g. `abp`, `ecg`, `central_venous_pressure`, `etco2`, `heart_rate`, …) plus
`timestamps_unix`, and the four top-level datasets.

---

## `/signals/` Group

Holds every per-sample 1-D array in the recording. Two kinds of dataset live
here: **physiologic channels** and the single **`timestamps_unix`** vector.
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

| Attribute           | Type             | Description                              |
|---------------------|------------------|------------------------------------------|
| `Fs`                | scalar `float64` | Sampling frequency in Hz (e.g. `200.0`). |
| `units`             | UTF-8 string     | Unit of measurement (e.g. `"mmHg"`, `"Volts"`, `"mL/min"`). May be an empty string. |
| `order_num`         | scalar `int64`   | The channel's display/order index in the original AcqKnowledge recording. Lets a reader reproduce the source channel ordering (channels are stored here by cleaned name, not in `order_num` order). |
| `point_count`       | scalar `int64`   | Number of samples in this channel's waveform. Equals the dataset length. For concatenated multi-file recordings this is the **final** length including `NaN` gap fill (recomputed after concatenation). |
| `frequency_divider` | scalar `int64`   | BIOPAC downsample factor for this channel relative to the file's native rate (`1` when the channel runs at the native rate). Provenance for diagnosing mixed-rate recordings; in files written by this converter all channels share one rate, so this is normally `1`. |
| `raw_scale_factor`  | scalar `float64` | Linear scale applied by BIOPAC to convert raw ADC integers to physical units: `physical = raw * raw_scale_factor + raw_offset`. Provenance only — see caveat below. |
| `raw_offset`        | scalar `float64` | Linear offset paired with `raw_scale_factor` (see formula above). |
| `dtype`             | UTF-8 string     | The channel's original numpy dtype in the `.acq` file, as a byte-order-tagged string (e.g. `">i2"` for raw 16-bit integer channels, `">f8"` for computed/float channels). |

`Fs` and `units` are always present; `units` reflects what was configured in
BIOPAC and may be outdated, so trust `Fs` over `units`. The six provenance
attributes (`order_num`, `point_count`, `frequency_divider`, `raw_scale_factor`,
`raw_offset`, `dtype`) are present on every file produced directly by
`acq2h5.py`. (`point_count` is always written. The other five may be absent only
in a file generated by `convert.py` from a *legacy* `.mat` that predates these
fields; a `.mat` produced by the current `acq2mat.py` carries all of them.)

> **`raw_scale_factor` / `raw_offset` caveat.** These describe how the *original*
> ADC integers mapped to physical units. The waveform stored here is already the
> **scaled** physical-unit `float64` data, and (for multi-file inputs) has had
> `NaN` gaps inserted. You therefore **cannot** losslessly reconstruct the raw
> integers from the stored waveform using these values — they are metadata about
> the source recording, not a round-trip key. On concatenation, the first file's
> scale/offset are kept; a per-channel `dtype` mismatch across files is warned
> about (not a hard error) and the first file's `dtype` is kept.

### `timestamps_unix` dataset

- **Path**: `/signals/timestamps_unix`.
- **Data**: 1-D `float64` array, one value per sample, aligned element-for-element
  with every channel dataset (same length).
- **Content**: **unix timestamps** — seconds since the Unix epoch
  (1970-01-01 00:00:00 UTC), as floats. These are *absolute instants*: the value
  is timezone-neutral (the local recording offset has already been folded in), so
  the same number means the same moment regardless of the reader's timezone. This
  makes the dataset the canonical key for cross-recording **alignment**.
- **Attributes**: `Fs` only (scalar `float64`) — *no* `units`. The `Fs` attribute
  exists so a "for every dataset in `signals/`, read `Fs`" loop works uniformly.

> `timestamps_unix` is deliberately stored *inside* `signals/` so a single read
> of the group yields every sample-aligned array (waveforms **and** the shared
> time axis) without a second fetch. When you want "just the physiologic
> channels," iterate `signals/` and skip the dataset named `timestamps_unix`.

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
  To index a 0-indexed Python/`h5py` array, subtract 1. To get a marker's
  absolute time, use `timestamps_unix[sample_index - 1]`.
- `seconds` / `minutes` are measured **from recording start** (sample 1), not
  from any wall clock.
- `time_local` uses the same **US/Eastern**, not-stored-in-file convention as
  `recording_start_local`; `date_created_utc` is the absolute (UTC) counterpart.
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
- **Timezone**: "local" means **US/Eastern** — the converter hardcodes this zone
  (the data originates at a US/Eastern site). The zone is **not** stored in the
  file, so the offset cannot be recovered from the string alone. For any
  timezone-correct computation, prefer the absolute sources: `timestamps_unix`
  (per sample) or `recording_start_utc` / `date_created_utc` (UTC with offset).
  The `_local` strings are display conveniences.

## `/Fs` Dataset

- **Type**: scalar `float64`.
- **Content**: the single sampling frequency (Hz) shared by all channels — a
  convenience accessor equal to each channel's `Fs` attribute.
- **Uniform-rate guarantee**: the format requires every channel to share one
  rate (this is what lets a single `timestamps_unix` axis and a single top-level
  `/Fs` describe the whole file). The converter **enforces** this: if channels
  disagree on `Fs` it raises an error and writes **no** file. Consequently, in
  every file that exists, top-level `/Fs` == each channel's `Fs` attribute.

---

## Root Attributes (file-level provenance)

File-level metadata is stored as HDF5 **attributes on the root group** (`/`),
not as datasets. Read them with `h5readatt(file, '/', name)` in MATLAB or
`f.attrs[name]` in Python.

| Attribute                    | Type             | Description                                              |
|------------------------------|------------------|---------------------------------------------------------|
| `file_revision`              | scalar `int64`   | The AcqKnowledge `.acq` file-format revision number from the graph header (e.g. `133`). Provenance for the source file version. For multi-file inputs this is the first file's value. |
| `native_samples_per_second`  | scalar `float64` | The file's native (authoritative) sampling rate in Hz. Normally equals top-level `/Fs` and each channel's `Fs`; preserved separately as the source-of-truth file rate. For multi-file inputs this is the first file's value. |

Both root attributes are written whenever the converter has file metadata
available (always, for files produced by `acq2h5.py`).

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

    # Channel provenance attributes
    order = f['signals/ecg'].attrs['order_num']         # original channel order
    dtype = f['signals/ecg'].attrs['dtype']             # '>i2'
    scale = f['signals/ecg'].attrs['raw_scale_factor']  # ADC->physical scale

    # File-level provenance (root attributes)
    revision = f.attrs['file_revision']                 # e.g. 133
    native_fs = f.attrs['native_samples_per_second']    # 200.0

    # Per-sample timestamps (unix seconds)
    ts = f['signals/timestamps_unix'][:]

    # All physiologic channel names (exclude the timestamp vector)
    channels = [k for k in f['signals'] if k != 'timestamps_unix']

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

% Channel provenance attributes
order = h5readatt('recording.h5', '/signals/ecg', 'order_num');
dtype = h5readatt('recording.h5', '/signals/ecg', 'dtype');  % '>i2'

% File-level provenance (root attributes)
revision  = h5readatt('recording.h5', '/', 'file_revision');
native_fs = h5readatt('recording.h5', '/', 'native_samples_per_second');

ts    = h5read('recording.h5', '/signals/timestamps_unix');
markers = jsondecode(h5read('recording.h5', '/event_markers'));

% unix seconds -> datetime
t = datetime(ts, 'ConvertFrom', 'epochtime', 'TicksPerSecond', 1);
```

A ready-made loader, `loadh5.m`, reads the whole file into a struct (signals,
per-signal `Fs`/`units`, timestamps, and an event-marker table).

---

## Aligning Recordings

`timestamps_unix` exists so that two signals — from the same file or from
different files/devices — can be placed on one common clock. Because the values
are absolute unix instants (not per-file sample offsets), alignment is just
matching timestamps.

```python
import h5py, numpy as np

def load_signal(path, name):
    with h5py.File(path, 'r') as f:
        return f[f'signals/{name}'][:], f['signals/timestamps_unix'][:]

ecg_a, t_a = load_signal('rec_a.h5', 'ecg')
abp_b, t_b = load_signal('rec_b.h5', 'abp')

# Resample/interpolate signal B onto signal A's timestamps (overlap only).
abp_on_a = np.interp(t_a, t_b, abp_b, left=np.nan, right=np.nan)
```

Notes for alignment:
- Timestamps are evenly spaced at `1/Fs` within a file, but **gaps** between
  concatenated source files appear as `NaN` runs in the waveforms while
  `timestamps_unix` keeps counting real wall-clock time across the gap. So a
  contiguous timestamp range can still contain `NaN` samples — check for them.
- Use `timestamps_unix` (absolute) for cross-file work; use `sample_index` /
  `seconds` (file-relative) only within a single recording.
- Do not align on the `_local` strings — they are display-only and carry no
  stored offset.

---

## Invariants (useful when validating or generating a file)

1. Every dataset under `/signals/` is 1-D `float64` and the **same length**.
2. Every `/signals/` dataset carries an `Fs` attribute; physiologic channels also
   carry `units` and `point_count`, plus — for files written directly by
   `acq2h5.py` — `order_num`, `frequency_divider`, `raw_scale_factor`,
   `raw_offset`, and `dtype` (these five may be absent only in a `convert.py`
   output built from a legacy `.mat`). `timestamps_unix` carries `Fs` but
   **not** `units` or the provenance attrs.
3. `len(timestamps_unix) == len(any channel)`.
4. Top-level `/Fs` equals every channel's `Fs` attribute. This always holds: the
   converter refuses to write a file when channels disagree on `Fs` (see the
   `/Fs` section), so a single shared rate is guaranteed.
5. `event_markers` decodes to an object of equal-length parallel arrays.
6. `sample_index` values are 1-indexed and lie in `[1, len(channel)]`.
7. Scalar string datasets (`event_markers`, `recording_start_utc`,
   `recording_start_local`) are stored as variable-length UTF-8; in Python they
   read back as `bytes` and should be `.decode('utf-8')`-ed.
8. `_local` time strings (`recording_start_local`, marker `time_local`) are in
   **US/Eastern**, which is **not** recorded in the file; absolute time comes
   from `timestamps_unix`, `recording_start_utc`, and marker `date_created_utc`.
9. The root group (`/`) carries `file_revision` (`int64`) and
   `native_samples_per_second` (`float64`) attributes. Each physiologic channel's
   `point_count` attribute equals that dataset's length.
