'''
acq2h5.py

Convert one or more ACQ files collected via Biopac's AcqKnowledge software to a
flat HDF5 (.h5) structure that is usable from both Python (h5py) and MATLAB
(h5read / h5readatt).

This is the HDF5 counterpart to acq2mat.py. It shares all parsing, multi-file
concatenation, NaN gap-filling, and event-marker logic with acq2mat.py via the
acq_common module; only the output stage differs.

Output layout
-------------
    /
    |-- signals/                  (HDF5 Group)
    |   |-- <signal_name>         (float64[] waveform; attrs: Fs, units)
    |   |-- ...
    |   |-- timestamps_local      (float64[] unix seconds; attr: Fs)
    |-- event_markers             (scalar UTF-8 string, JSON-encoded)
    |-- recording_start_utc       (scalar UTF-8 string, ISO)
    |-- recording_start_local     (scalar UTF-8 string, 'YYYY-MM-DD HH:MM:SS.mmm')
    |-- Fs                        (scalar float64 -- shared sampling rate)

Design notes
------------
- Each signal is reachable by name alone: signals/<name> IS the waveform array.
  Sampling rate and units live as HDF5 attributes on that dataset, so the
  single-depth access rule holds in both Python and MATLAB.
      Python : f['signals/ecg'][:]              ; f['signals/ecg'].attrs['Fs']
      MATLAB : h5read(file, '/signals/ecg')     ; h5readatt(file, '/signals/ecg', 'Fs')
- timestamps_local is stored INSIDE signals/ (not at the top level), and carries
  an Fs attribute so a "for every dataset in signals, read Fs" loop works
  uniformly. Values are unix timestamps (float64 seconds since 1970-01-01 UTC),
  the same convention as acq2mat.py -- NOT MATLAB datenums.
      MATLAB : datetime(ts, 'ConvertFrom', 'epochtime', 'TicksPerSecond', 1)
- This output deliberately diverges from biopac_h5_format.md (timestamps inside
  signals/, unix seconds rather than datenums).
'''

import sys
import argparse
import os
import json
import bioread
import numpy as np
import pytz
import h5py

from acq_common import (
    parse_data,
    cat_multiple_files,
    validate_sampling_rates,
    build_timestamp_vector,
    export_event_markers_csv,
)

# Top-level keys in the processing dict that are NOT physiologic channels.
_NON_SIGNAL_KEYS = {
    'event_markers',
    'recording_start_utc',
    'recording_start_local',
    'Fs',
    'timestamps_local',
}


def argument_parser(argv):
    '''Parse input from the command line'''

    parser = argparse.ArgumentParser(description='ACQ2H5: a tool to extract ACQ files to HDF5.')

    parser.add_argument('file',
        help='ACQ file to convert',
        nargs='+')

    parser.add_argument('-o', '--outfile',
        help='Filename for HDF5 file output',
        required=False)

    args = parser.parse_args()

    if not args.outfile:
        args.outfile = args.file[0].replace('.acq', '.h5')

    return args


def write_h5(d, outfile):
    '''Write the processed data dictionary to a flat HDF5 file.

    Args:
        d: data dictionary produced by parse_data/cat_multiple_files, augmented
           in __main__ with 'recording_start_utc', 'recording_start_local',
           'Fs', and 'timestamps_local'. Channel entries are dicts with
           'wave', 'Fs', and 'unit'. 'event_markers' has already had its
           datetime values converted to ISO strings.
        outfile: path to the .h5 file to create (overwritten if it exists).
    '''
    str_dtype = h5py.string_dtype('utf-8')

    # Identify physiologic channels (everything that isn't bookkeeping).
    channel_keys = [k for k in d.keys() if k not in _NON_SIGNAL_KEYS]

    with h5py.File(outfile, 'w') as f:
        signals = f.create_group('signals')

        # Each signal: dataset named by signal, Fs/units as attributes.
        for name in channel_keys:
            wave = np.asarray(d[name]['wave'], dtype=np.float64)
            ds = signals.create_dataset(name, data=wave, compression='gzip')
            ds.attrs['Fs'] = np.float64(d[name]['Fs'])
            # Store units as a UTF-8 string attribute (may be empty). Writing a
            # Python str with an explicit utf-8 dtype keeps it a scalar string
            # that MATLAB's h5readatt reads as a char/string.
            units = d[name].get('unit') or ''
            ds.attrs.create('units', units, dtype=str_dtype)

        # timestamps_local lives inside signals/, with an Fs attribute so that
        # iterating signals uniformly still finds an Fs on every dataset.
        ts = np.asarray(d['timestamps_local'], dtype=np.float64)
        ts_ds = signals.create_dataset('timestamps_local', data=ts, compression='gzip')
        ts_ds.attrs['Fs'] = np.float64(d['Fs'])

        # event_markers: one scalar JSON string (datetimes already ISO strings).
        em_json = json.dumps(d['event_markers'])
        f.create_dataset('event_markers', data=em_json, dtype=str_dtype)

        # Top-level scalar metadata.
        f.create_dataset('recording_start_utc', data=d['recording_start_utc'], dtype=str_dtype)
        f.create_dataset('recording_start_local', data=d['recording_start_local'], dtype=str_dtype)
        f.create_dataset('Fs', data=np.float64(d['Fs']))


if __name__ == '__main__':

    args = argument_parser(sys.argv[1:])
    data = [bioread.read_file(i) for i in args.file] # read each file specified in command line

    # Parse data and extract start times
    parsed = [parse_data(i) for i in data]
    d_list = [p[0] for p in parsed]
    start_times = [p[1] for p in parsed]

    if len(d_list) >= 2: # concatenate files if there are more than one
        d = cat_multiple_files(d_list, start_times)
    else:
        d = d_list[0]

    # Validate that all channels share the same sampling frequency
    validate_sampling_rates(d)

    # Add metadata for time vector calculation (always, for both single and multi-file)
    channel_keys = [k for k in d.keys() if k != 'event_markers']
    Fs = d[channel_keys[0]]['Fs']
    d['recording_start_utc'] = start_times[0].isoformat()
    EST = pytz.timezone('US/Eastern')
    start_time_local = start_times[0].astimezone(EST)
    d['recording_start_local'] = start_time_local.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    d['Fs'] = Fs

    # Build local timestamp vector (unix seconds) for all samples
    n_samples = len(d[channel_keys[0]]['wave'])
    d['timestamps_local'] = build_timestamp_vector(start_time_local, Fs, n_samples)

    # Export event markers to CSV (before wrapping and saving to HDF5)
    csv_date = start_time_local.strftime('%Y-%m-%d')
    csv_output_dir = os.path.dirname(args.outfile)
    csv_output_path = os.path.join(csv_output_dir, f'{csv_date}_extracted_comments.csv') if csv_output_dir else f'{csv_date}_extracted_comments.csv'
    try:
        export_event_markers_csv(d['event_markers'], csv_output_path)
        print(f"Event markers exported to: {csv_output_path}")
    except ValueError as e:
        print(f"ERROR: Cannot export event markers CSV: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not export CSV: {e}")

    # Convert datetime objects to ISO strings for JSON/HDF5 compatibility
    if 'date_created_utc' in d['event_markers']:
        d['event_markers']['date_created_utc'] = [
            dt.isoformat() for dt in d['event_markers']['date_created_utc']
        ]

    write_h5(d, args.outfile)
    print(f"HDF5 file saved to: {args.outfile}")
