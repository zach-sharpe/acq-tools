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
    |   |-- timestamps_unix       (float64[] unix seconds; attr: Fs)
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
- timestamps_unix is stored INSIDE signals/ (not at the top level), and carries
  an Fs attribute so a "for every dataset in signals, read Fs" loop works
  uniformly. Values are unix timestamps (float64 seconds since 1970-01-01 UTC),
  the same convention as acq2mat.py -- NOT MATLAB datenums. Storing the
  timestamps alongside the signals means a single read of signals/ yields every
  sample-aligned array (waveforms + the shared time axis) without a second
  fetch -- which matters for alignment scripts.
      MATLAB : datetime(ts, 'ConvertFrom', 'epochtime', 'TicksPerSecond', 1)
- This output deliberately diverges from biopac_h5_format.md (timestamps inside
  signals/, unix seconds rather than datenums).
'''

import sys
import argparse
import os
import json
import numpy as np
import h5py

# NOTE: this module is imported by convert_acq (for write_h5 / _NON_SIGNAL_KEYS)
# and by convert.py. To keep those imports lightweight and avoid a circular
# import, the conversion pipeline lives in convert_acq -- this file only owns
# the HDF5 writer plus a thin CLI (which imports convert_acq lazily in __main__).

# Top-level keys in the processing dict that are NOT physiologic channels.
_NON_SIGNAL_KEYS = {
    'event_markers',
    'recording_start_utc',
    'recording_start_local',
    'Fs',
    'timestamps_unix',
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


def write_h5(d, outfile, file_meta=None):
    '''Write the processed data dictionary to a flat HDF5 file.

    Args:
        d: data dictionary produced by parse_data/cat_multiple_files, augmented
           in __main__ with 'recording_start_utc', 'recording_start_local',
           'Fs', and 'timestamps_unix'. Channel entries are dicts with
           'wave', 'Fs', 'unit', and channel provenance ('order_num',
           'point_count', 'frequency_divider', 'raw_scale_factor',
           'raw_offset', 'dtype'). 'event_markers' has already had its
           datetime values converted to ISO strings.
        outfile: path to the .h5 file to create (overwritten if it exists).
        file_meta: optional dict of file-level provenance ('file_revision',
           'native_samples_per_second') written as root-level HDF5 attributes.
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

            # Channel provenance attributes (see acq2h5_format.md). Numeric
            # attrs are stored with explicit numpy types; dtype is a string.
            # Each is written only when present, so write_h5 also accepts
            # legacy inputs (e.g. old .mat files via convert.py) that lack them.
            # point_count falls back to the actual waveform length.
            chan = d[name]
            if 'order_num' in chan:
                ds.attrs['order_num'] = np.int64(chan['order_num'])
            ds.attrs['point_count'] = np.int64(chan.get('point_count', len(wave)))
            if 'frequency_divider' in chan:
                ds.attrs['frequency_divider'] = np.int64(chan['frequency_divider'])
            if 'raw_scale_factor' in chan:
                ds.attrs['raw_scale_factor'] = np.float64(chan['raw_scale_factor'])
            if 'raw_offset' in chan:
                ds.attrs['raw_offset'] = np.float64(chan['raw_offset'])
            if 'dtype' in chan:
                ds.attrs.create('dtype', chan['dtype'], dtype=str_dtype)

        # timestamps_unix lives inside signals/, with an Fs attribute so that
        # iterating signals uniformly still finds an Fs on every dataset.
        ts = np.asarray(d['timestamps_unix'], dtype=np.float64)
        ts_ds = signals.create_dataset('timestamps_unix', data=ts, compression='gzip')
        ts_ds.attrs['Fs'] = np.float64(d['Fs'])

        # event_markers: one scalar JSON string (datetimes already ISO strings).
        em_json = json.dumps(d['event_markers'])
        f.create_dataset('event_markers', data=em_json, dtype=str_dtype)

        # Top-level scalar metadata.
        f.create_dataset('recording_start_utc', data=d['recording_start_utc'], dtype=str_dtype)
        f.create_dataset('recording_start_local', data=d['recording_start_local'], dtype=str_dtype)
        f.create_dataset('Fs', data=np.float64(d['Fs']))

        # File-level provenance as root attributes (MATLAB: h5readatt(file, '/', name)).
        # Each is written only when present, so legacy inputs (e.g. an old .mat
        # via convert.py) that lack file metadata simply omit these attrs.
        if file_meta:
            if 'file_revision' in file_meta:
                f.attrs['file_revision'] = np.int64(file_meta['file_revision'])
            if 'native_samples_per_second' in file_meta:
                f.attrs['native_samples_per_second'] = np.float64(file_meta['native_samples_per_second'])


if __name__ == '__main__':

    # Imported lazily (only when run as a script) to avoid a circular import:
    # convert_acq imports write_h5 from this module.
    from convert_acq import convert_acq

    args = argument_parser(sys.argv[1:])

    # convert_acq() takes (folder, filename); the CLI exposes a single -o path.
    # Split it back apart, mapping an empty directory to the current directory.
    output_folder, output_filename = os.path.split(args.outfile)
    output_folder = output_folder or '.'

    try:
        result = convert_acq(args.file, output_folder, output_filename, fmt='h5')
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for msg in result.messages:
        print(msg)
    print(f"HDF5 file saved to: {result.output_path}")
