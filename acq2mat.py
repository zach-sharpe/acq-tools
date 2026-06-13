'''
acq2mat.py

Convert an ACQ file collected via Biopac's AcqKnowledge software to a MATLAB structure.
'''

import sys
import argparse
import os
import bioread
import pytz
from scipy import io as sio

from acq_common import (
    clean,
    load_signal_renaming,
    timestamps_to_datetime,
    validate_sampling_rates,
    build_timestamp_vector,
    parse_data,
    export_event_markers_csv,
    cat_multiple_files,
)

def argument_parser(argv):
    '''Parse input from the command line'''

    parser = argparse.ArgumentParser(description='ACQ2MAT: a tool to extract ACQ files.')

    parser.add_argument('file',
        help='ACQ file to convert',
        nargs='+')

    parser.add_argument('-o', '--outfile',
        help='Filename for MATLAB file output',
        required=False)

    args = parser.parse_args()

    if not args.outfile:
        args.outfile = args.file[0].replace('.acq', '.mat')

    return args


if __name__ == '__main__':

    args = argument_parser(sys.argv[1:])
    data = [bioread.read_file(i) for i in args.file] # read each file specified in command line

    # Parse data and extract start times
    parsed = [parse_data(i) for i in data]
    d_list = [p[0] for p in parsed]
    start_times = [p[1] for p in parsed]
    file_meta = parsed[0][2]  # file-level provenance from the first file

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

    # File-level provenance (channel-level provenance already lives inside each
    # channel struct via parse_data). Added after channel_keys is computed so it
    # is not mistaken for a channel.
    d['file_revision'] = file_meta['file_revision']
    d['native_samples_per_second'] = file_meta['native_samples_per_second']

    # Build local timestamp vector (MATLAB datenums) for all samples
    n_samples = len(d[channel_keys[0]]['wave'])
    d['timestamps_local'] = build_timestamp_vector(start_time_local, Fs, n_samples)

    # Export event markers to CSV (before wrapping and saving to MAT)
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

    # Convert datetime objects to ISO strings for MATLAB compatibility
    if 'date_created_utc' in d['event_markers']:
        d['event_markers']['date_created_utc'] = [
            dt.isoformat() for dt in d['event_markers']['date_created_utc']
        ]

    d = {'d': d} # wrap into one MATLAB struct rather than multiple variables

    sio.savemat(args.outfile, d, oned_as='column', do_compression=True)
    print(f"MAT file saved to: {args.outfile}")
