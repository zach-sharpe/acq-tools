# MAT File Output Structure

Output from `acq2mat.py`. All data is stored under a single MATLAB struct variable **`d`**.

```
d
├── <channel_name>          (one per ACQ channel, e.g. d.abp, d.ecg)
│   ├── .wave               n-by-1 double  — waveform data
│   ├── .Fs                 scalar double   — sampling frequency (Hz)
│   └── .unit               char            — unit of measure (from ACQ config)
│
├── event_markers           struct (use loadfile.m to convert to table)
│   ├── .label              n-by-1 cellstr  — annotation text
│   ├── .sample_index       n-by-1 double   — 1-indexed sample number
│   ├── .type_code          n-by-1 cellstr  — internal ACQ notation
│   ├── .type               n-by-1 cellstr  — marker creation method
│   ├── .channel_number     n-by-1 double   — associated channel (often empty)
│   ├── .channel            n-by-1 cellstr  — associated channel name (often empty)
│   ├── .seconds            n-by-1 double   — event time in seconds from start
│   └── .minutes            n-by-1 double   — event time in minutes from start
│
├── timestamps_local        n-by-1 double   — unix timestamps for every sample (seconds since epoch)
├── Fs                      scalar double   — global sampling frequency (Hz)
├── recording_start_utc     char            — ISO 8601 string (UTC)
└── recording_start_local   char            — 'YYYY-MM-DD HH:MM:SS.mmm' (US/Eastern)
```

## Field Details

### Channel fields (`d.<channel_name>`)

Each ACQ channel becomes a field on `d`. Channel names are cleaned to valid MATLAB variable names (lowercase, alphanumeric + underscore, must start with a letter). Optional renaming is applied via `signal_renaming.json`.

| Subfield | Type | Description |
|----------|------|-------------|
| `wave` | n-by-1 double | Raw waveform samples. For concatenated files, segments are joined with NaN-filled gaps. |
| `Fs` | scalar double | Sampling frequency in Hz. All channels must share the same Fs. |
| `unit` | char | Unit string from AcqKnowledge (may be outdated). |

### `event_markers`

Saved as a struct by `scipy.io.savemat`. Use `loadfile.m` to convert to a MATLAB table:

```matlab
d = loadfile('recording.mat');
d.event_markers  % now a table
```

| Column | Type | Description |
|--------|------|-------------|
| `label` | cellstr | Annotation text (e.g., "Intubation", "Drug given") |
| `sample_index` | double | 1-indexed sample number in the waveform vectors |
| `type_code` | cellstr | Internal ACQ type code (events with `'nrto'` are excluded) |
| `type` | cellstr | Human-readable marker type (e.g., "User Type 9" for F9 key) |
| `channel_number` | double | Channel associated with the marker (often NaN) |
| `channel` | cellstr | Channel name associated with the marker (often empty) |
| `seconds` | double | Time of event in seconds from recording start |
| `minutes` | double | Time of event in minutes from recording start |

### `timestamps_local`

A timestamp for every sample in the recording, stored as **unix timestamps** (float64 seconds since 1970-01-01 00:00:00 UTC). This vector has the same length as each channel's `wave` field.

Timestamps reflect real wall-clock time. For concatenated files, timestamps increment continuously through NaN gaps so that the time axis always corresponds to actual clock time.

**Convert to MATLAB datetime:**
```matlab
d = loadfile('recording.mat');
t = datetime(d.timestamps_local, 'ConvertFrom', 'epochtime', 'TicksPerSecond', 1);
```

**Plot a channel against wall-clock time:**
```matlab
d = loadfile('recording.mat');
t = datetime(d.timestamps_local, 'ConvertFrom', 'epochtime', 'TicksPerSecond', 1);
plot(t, d.abp.wave)
xlabel('Time')
ylabel(sprintf('ABP (%s)', d.abp.unit))
```

**Convert to Python datetime:**
```python
from acq2mat import timestamps_to_datetime
from scipy.io import loadmat

mat = loadmat('recording.mat', squeeze_me=True)
d = mat['d'][()]
t = timestamps_to_datetime(d['timestamps_local'])
```

**Convert to Pandas DatetimeIndex:**
```python
import pandas as pd
from scipy.io import loadmat

mat = loadmat('recording.mat', squeeze_me=True)
d = mat['d'][()]
t = pd.to_datetime(d['timestamps_local'], unit='s', utc=True).tz_convert('US/Eastern')
```

### `Fs`

Top-level scalar double. Global sampling frequency in Hz (all channels are validated to share the same rate). Redundant with per-channel `Fs` but provided for convenience.

### `recording_start_utc`

ISO 8601 formatted string of the recording start time in UTC. Derived from the earliest event marker timestamp in the ACQ file.

Example: `'2024-06-15T18:30:00.123456+00:00'`

### `recording_start_local`

Recording start time converted to US/Eastern timezone.

Example: `'2024-06-15 14:30:00.123'`

## Concatenated Files

When multiple ACQ files are passed to `acq2mat.py`, they are joined in order:

- Channel waveforms are concatenated with **NaN-filled gaps** spanning the real time between recordings
- Event marker `sample_index` values are offset to reflect position in the combined vector
- `timestamps_local` increments continuously through gaps (wall-clock time is preserved)
- `recording_start_*` fields refer to the start of the **first** file

## Companion Files

| File | Description |
|------|-------------|
| `*_events.csv` | CSV export of event markers with a `time (EST)` column. Generated alongside the MAT file. |
